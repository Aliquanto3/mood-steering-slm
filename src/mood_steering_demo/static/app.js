"use strict";
// Interface de la démo : trois colonnes alimentées par un flux SSE (GET /stream), lu par
// fetch + ReadableStream (et non EventSource) : le code HTTP (200, 409, 422) est connu sans
// seconde requête, et aucune reconnexion automatique ne relance de génération.
// Texte : expression d'humeur (ton du texte) uniquement.

const ARMS = ["neutral", "prompt_joyful", "steering"];
const FALLBACK_TEXT = "pas de vecteur pour ce modèle : repli sur le prompt";
const $ = (id) => document.getElementById(id);
const STOP_GRACE_MS = 2000; // un 409 dans ce délai après Arrêter : verrou serveur pas encore rendu
const RETRY_DELAY_MS = 500;
let controller = null; // AbortController de la génération en cours
let config = null;
let doseApplied = 0;
let fallbackActive = false;
let sawDone = false;
let lastStopAt = 0;
let curvePending = false;
let steeringTokens = 0;
let doseHistory = [];
let pulseActive = false; // le flux porte la dose appliquée à chaque token (profil impulsion)

function formatDose(v) {
  return Number(v).toFixed(2).replace(".", ",");
}

function setStatus(message, isError) {
  const el = $("status");
  el.textContent = message;
  el.className = isError ? "error" : "";
}

function setRunning(running) {
  $("generate").disabled = running;
  $("stop").disabled = !running;
}

/**
 * Trace la courbe de dose dans le SVG #dose-curve.
 * @param {Array<{x:number, dose:number}>} points un point par token (x = rang du token, à partir
 *   de 0 ; dose dans [-1, 1]). Profil constant : la dose du curseur ; profil impulsion : la dose
 *   réellement appliquée à chaque token, lue dans le flux (story 5.1). Une liste vide efface la courbe (axes et repère 0
 *   restent affichés).
 */
function drawDoseCurve(points) {
  const svg = $("dose-curve");
  const W = 400, H = 140, L = 34, R = 10, T = 10, B = 24;
  const ns = "http://www.w3.org/2000/svg";
  const yOf = (d) => T + ((1 - d) / 2) * (H - T - B);
  const maxX = Math.max(1, ...points.map((p) => p.x));
  const xOf = (x) => L + (x / maxX) * (W - L - R);
  const el = (name, attrs, text) => {
    const e = document.createElementNS(ns, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  svg.replaceChildren();
  svg.append(
    el("line", { class: "axis", x1: L, y1: T, x2: L, y2: H - B }),
    el("line", { class: "axis", x1: L, y1: H - B, x2: W - R, y2: H - B }),
    el("line", { class: "zero", x1: L, y1: yOf(0), x2: W - R, y2: yOf(0) }),
    el("text", { x: 4, y: yOf(1) + 4 }, "+1"),
    el("text", { x: 4, y: yOf(0) + 4 }, "0"),
    el("text", { x: 4, y: yOf(-1) + 4 }, "−1"),
    el("text", { x: L, y: H - 8 }, "0"),
    el("text", { x: W - R - 60, y: H - 8 }, "tokens"),
  );
  if (points.length === 0) return;
  if (fallbackActive) {
    svg.append(el("text", { class: "legend", x: W - R, y: T + 14, "text-anchor": "end" }, "repli : dose sans effet"));
  }
  const d = points.map((p, i) => `${i ? "L" : "M"}${xOf(p.x).toFixed(1)} ${yOf(p.dose).toFixed(1)}`);
  if (points.length === 1) d.push(`L${xOf(1).toFixed(1)} ${yOf(points[0].dose).toFixed(1)}`);
  svg.append(el("path", { class: fallbackActive ? "line fallback" : "line", d: d.join(" ") }));
}

// Un tracé par image au plus, quel que soit le débit de tokens (sinon O(n²) sur la génération).
function scheduleCurve() {
  if (curvePending) return;
  curvePending = true;
  requestAnimationFrame(() => {
    curvePending = false;
    drawDoseCurve(doseHistory);
  });
}

function resetColumns() {
  for (const arm of ARMS) {
    $("out-" + arm).textContent = "";
    const b = $("badge-" + arm);
    b.hidden = true;
    b.textContent = "";
    b.classList.remove("error");
  }
  steeringTokens = 0;
  doseHistory = [];
  fallbackActive = false;
  pulseActive = false;
  drawDoseCurve([]);
}

function showBadge(arm, text, isError) {
  const b = $("badge-" + arm);
  b.textContent = text;
  b.classList.toggle("error", Boolean(isError));
  b.hidden = false;
}

function finishStream() {
  controller = null;
  setRunning(false);
}

function handleEvent(ev) {
  if (ev.type === "start") {
    if (ev.arm === "steering") {
      doseApplied = ev.dose;
      pulseActive = ev.profile === "pulse";
      fallbackActive = Boolean(ev.fallback);
      if (fallbackActive) showBadge("steering", FALLBACK_TEXT, false);
    }
  } else if (ev.type === "token") {
    const out = $("out-" + ev.arm);
    if (!out) return;
    out.append(ev.text); // nœud texte ajouté : pas de recopie du texte déjà affiché
    if (ev.arm === "steering") {
      // En repli, la dose n'agit pas : on trace 0 (et la légende le dit).
      // Impulsion : dose du flux pour ce token ; constant : celle du curseur (inchangé).
      const dose = pulseActive && typeof ev.dose === "number" ? ev.dose : doseApplied;
      doseHistory.push({ x: steeringTokens++, dose: fallbackActive ? 0 : dose });
      scheduleCurve();
    }
  } else if (ev.type === "error") {
    if ($("badge-" + ev.arm)) showBadge(ev.arm, "Erreur : " + ev.message, true);
  } else if (ev.type === "done") {
    sawDone = true;
    finishStream();
    setStatus("Génération terminée.", false);
  }
}

// Une trame SSE : lignes « data: ... » (les autres champs sont ignorés), terminée par une ligne vide.
function dispatchFrame(frame) {
  const data = frame
    .split("\n")
    .map((l) => (l.endsWith("\r") ? l.slice(0, -1) : l))
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(l.startsWith("data: ") ? 6 : 5))
    .join("\n");
  if (data) handleEvent(JSON.parse(data));
}

async function refusalMessage(response) {
  let detail = null;
  try {
    detail = (await response.json()).detail;
  } catch (e) {
    detail = null;
  }
  const code = response.status;
  if (code === 409) return "Une génération est déjà en cours : attendez qu'elle se termine.";
  if (code === 422) {
    const why = typeof detail === "string" ? detail : "prompt vide ou trop long, ou dose hors de [−1 ; +1]";
    return "Paramètres refusés : " + why + ".";
  }
  return "Erreur du serveur (" + code + ").";
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runStream(params, retried) {
  const ctrl = new AbortController();
  controller = ctrl;
  sawDone = false;
  let response;
  try {
    response = await fetch("/stream?" + params, { signal: ctrl.signal, headers: { Accept: "text/event-stream" } });
  } catch (e) {
    if (controller !== ctrl) return; // arrêté par l'utilisateur
    finishStream();
    setStatus("Serveur injoignable.", true);
    return;
  }
  if (controller !== ctrl) return;
  if (response.status === 409 && !retried && Date.now() - lastStopAt < STOP_GRACE_MS) {
    // Juste après Arrêter, le serveur libère son verrou un peu plus tard : un seul nouvel essai.
    setStatus("Génération précédente en cours d'arrêt, nouvel essai…", false);
    await sleep(RETRY_DELAY_MS);
    if (controller !== ctrl) return;
    return runStream(params, true);
  }
  if (!response.ok || !response.body) {
    const message = await refusalMessage(response);
    if (controller !== ctrl) return;
    finishStream();
    setStatus(message, true);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        dispatchFrame(frame);
      }
    }
  } catch (e) {
    if (controller !== ctrl) return; // AbortError après Arrêter
  }
  if (controller === ctrl && !sawDone) {
    finishStream();
    setStatus("Connexion interrompue.", true);
  }
}

function generate(event) {
  event.preventDefault();
  if (controller) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    setStatus("Saisissez un prompt.", true);
    return;
  }
  resetColumns();
  const params = new URLSearchParams({ prompt, dose: String($("dose").value) });
  if (config && config.max_tokens_limit) {
    params.set("max_tokens", String(Math.min(128, config.max_tokens_limit)));
  }
  if ($("profile").value === "pulse") {
    params.set("profile", "pulse");
    params.set("rise", String($("rise").value));
    params.set("half_life", String($("half-life").value));
  }
  setStatus("Génération en cours…", false);
  setRunning(true);
  runStream(params, false);
}

function stop() {
  if (!controller) return;
  const ctrl = controller;
  finishStream();
  ctrl.abort();
  lastStopAt = Date.now();
  setStatus("Génération arrêtée.", false);
}

async function init() {
  $("form").addEventListener("submit", generate);
  $("stop").addEventListener("click", stop);
  $("dose").addEventListener("input", () => {
    $("dose-value").textContent = formatDose($("dose").value);
  });
  $("dose-value").textContent = formatDose($("dose").value);
  $("profile").addEventListener("change", () => {
    $("pulse-params").hidden = $("profile").value !== "pulse";
  });
  drawDoseCurve([]);
  try {
    config = await (await fetch("/config")).json();
    for (const arm of ARMS) {
      if (config.labels && config.labels[arm]) $("title-" + arm).textContent = config.labels[arm];
    }
    const dose = $("dose");
    dose.min = config.dose_min;
    dose.max = config.dose_max;
    if (config.max_prompt_chars) $("prompt").maxLength = config.max_prompt_chars;
    if (config.has_vector === false) showBadge("steering", FALLBACK_TEXT, false);
  } catch (e) {
    setStatus("Configuration indisponible : serveur injoignable.", true);
  }
}

init();
