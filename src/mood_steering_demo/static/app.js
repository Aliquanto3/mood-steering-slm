"use strict";
// Interface de la démo : trois colonnes alimentées par un flux SSE (GET /stream).
// Texte : expression d'humeur (ton du texte) uniquement.

const ARMS = ["neutral", "prompt_joyful", "steering"];
const FALLBACK_TEXT = "pas de vecteur pour ce modèle : repli sur le prompt";
const $ = (id) => document.getElementById(id);
let source = null;
let config = null;
let doseApplied = 0;
let steeringTokens = 0;
let doseHistory = [];

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
 *   de 0 ; dose dans [-1, 1]). Pour l'instant la dose est constante ; un profil token par token
 *   (story 2.3) pourra être passé tel quel. Une liste vide efface la courbe (axes et repère 0
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
  const d = points.map((p, i) => `${i ? "L" : "M"}${xOf(p.x).toFixed(1)} ${yOf(p.dose).toFixed(1)}`);
  if (points.length === 1) d.push(`L${xOf(1).toFixed(1)} ${yOf(points[0].dose).toFixed(1)}`);
  svg.append(el("path", { class: "line", d: d.join(" ") }));
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
  drawDoseCurve([]);
}

function showBadge(arm, text, isError) {
  const b = $("badge-" + arm);
  b.textContent = text;
  b.classList.toggle("error", Boolean(isError));
  b.hidden = false;
}

function closeSource() {
  if (source) {
    source.close();
    source = null;
  }
  setRunning(false);
}

function handleEvent(ev) {
  if (ev.type === "start") {
    if (ev.arm === "steering") {
      doseApplied = ev.dose;
      if (ev.fallback) showBadge("steering", FALLBACK_TEXT, false);
    }
  } else if (ev.type === "token") {
    const out = $("out-" + ev.arm);
    if (!out) return;
    out.textContent += ev.text;
    if (ev.arm === "steering") {
      doseHistory.push({ x: steeringTokens++, dose: doseApplied });
      drawDoseCurve(doseHistory);
    }
  } else if (ev.type === "error") {
    showBadge(ev.arm, "Erreur : " + ev.message, true);
  } else if (ev.type === "done") {
    closeSource();
    setStatus("Génération terminée.", false);
  }
}

async function explainRefusal(params) {
  // EventSource ne donne pas le code HTTP : on relit le statut de /stream pour le message.
  try {
    const r = await fetch("/stream?" + params);
    const code = r.status;
    if (r.body) r.body.cancel();
    if (code === 409) return "Une génération est déjà en cours : attendez qu'elle se termine.";
    if (code === 422) return "Paramètres refusés : prompt vide ou trop long, ou dose hors de [−1 ; +1].";
    if (code === 200) return "Connexion interrompue.";
    return "Erreur du serveur (" + code + ").";
  } catch (e) {
    return "Serveur injoignable.";
  }
}

function generate(event) {
  event.preventDefault();
  if (source) return;
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
  setStatus("Génération en cours…", false);
  setRunning(true);
  const es = new EventSource("/stream?" + params);
  source = es;
  es.onmessage = (m) => handleEvent(JSON.parse(m.data));
  es.onerror = async () => {
    // Le flux se ferme après "done" (déjà traité) ; ici, c'est un refus ou une coupure.
    if (source !== es) return;
    closeSource();
    setStatus(await explainRefusal(params), true);
  };
}

function stop() {
  if (!source) return;
  closeSource();
  setStatus("Génération arrêtée.", false);
}

async function init() {
  $("form").addEventListener("submit", generate);
  $("stop").addEventListener("click", stop);
  $("dose").addEventListener("input", () => {
    $("dose-value").textContent = formatDose($("dose").value);
  });
  $("dose-value").textContent = formatDose($("dose").value);
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
