const state = {
  sessionId: null,
  textures: [],
  previews: { env: "", raw: "", floor: "" },
  activeTab: "floor",
};

const ui = {
  imageInput: document.getElementById("imageInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusText: document.getElementById("statusText"),
  previewImg: document.getElementById("previewImg"),
  previewTabs: document.getElementById("previewTabs"),
  previewLegend: document.getElementById("previewLegend"),
  resultImg: document.getElementById("resultImg"),
  texturesGrid: document.getElementById("texturesGrid"),
};

const TAB_LEGENDS = {
  env: "Rojo: capa IA de lo que NO es piso (cielo, césped, muebles…). No es el color real de la foto.",
  raw: "Cyan: capa IA del piso antes de quitar muebles/entorno.",
  floor: "Verde: capa IA final solo piso. Si ves sillones verdes aquí, la máscara aún los incluye — revisa pestaña Entorno.",
};

function setActiveTab(tab) {
  state.activeTab = tab;
  const src = state.previews[tab];
  if (src) ui.previewImg.src = src;
  ui.previewLegend.textContent = TAB_LEGENDS[tab] || "";
  document.querySelectorAll(".preview-tab").forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("border-sky-600", active);
    btn.classList.toggle("bg-sky-50", active);
    btn.classList.toggle("font-medium", active);
  });
}

document.querySelectorAll(".preview-tab").forEach((btn) => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
});

async function fetchTextures() {
  const res = await fetch("/api/textures");
  if (!res.ok) throw new Error("No se pudo cargar el catalogo");
  const data = await res.json();
  state.textures = data.textures || [];
  ui.texturesGrid.innerHTML = "";
  if (!state.textures.length) {
    ui.texturesGrid.innerHTML = `<p class="text-sm text-slate-500">No hay materiales disponibles.</p>`;
    return;
  }
  state.textures.forEach((item) => {
    const card = document.createElement("button");
    card.className = "rounded-xl border p-2 text-left hover:border-sky-500";
    card.innerHTML = `
      <img src="${item.image_url}" alt="${item.name}" class="h-24 w-full rounded-lg object-cover" />
      <p class="mt-2 text-sm font-semibold">${item.name}</p>
      <p class="text-xs text-slate-500">${item.category || "general"}</p>
    `;
    card.addEventListener("click", () => applyTexture(item.id));
    ui.texturesGrid.appendChild(card);
  });
}

async function analyze() {
  const file = ui.imageInput.files?.[0];
  if (!file) {
    alert("Selecciona o toma una foto primero.");
    return;
  }
  ui.statusText.textContent = "IA: entorno + muebles + piso… (~1–2 min). El verde es solo la capa de vista previa.";
  ui.previewTabs.classList.add("hidden");
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch("/api/analyze", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "No se pudo analizar la imagen");
  state.sessionId = data.session_id;

  const b64 = (s) => (s ? `data:image/jpeg;base64,${s}` : "");
  state.previews.floor = b64(data.mask_preview_base64);
  state.previews.env = b64(data.environment_preview_base64);
  state.previews.raw = b64(data.raw_floor_preview_base64) || state.previews.floor;

  ui.previewTabs.classList.remove("hidden");
  ui.previewLegend.classList.remove("hidden");
  const envBtn = document.querySelector('.preview-tab[data-tab="env"]');
  if (envBtn) envBtn.style.display = data.environment_detected ? "" : "none";
  setActiveTab("floor");

  const msg = data.message || "Analisis listo. Revisa las 3 capas y elige material.";
  ui.statusText.textContent = msg;
  ui.statusText.className = msg.includes("fallback") ? "mt-3 text-sm text-amber-600" : "mt-3 text-sm text-slate-600";
}

async function applyTexture(textureId) {
  if (!state.sessionId) {
    alert("Primero analiza una foto.");
    return;
  }
  ui.statusText.textContent = "Aplicando textura...";
  const res = await fetch("/api/visualize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, texture_id: textureId }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "No se pudo aplicar textura");
  ui.resultImg.src = `data:image/jpeg;base64,${data.result_image_base64}`;
  ui.statusText.textContent = "Resultado listo.";
}

ui.analyzeBtn.addEventListener("click", async () => {
  try {
    await analyze();
  } catch (err) {
    ui.statusText.textContent = String(err.message || err);
  }
});

fetchTextures().catch((err) => {
  ui.statusText.textContent = String(err.message || err);
});
