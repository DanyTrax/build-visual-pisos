const state = {
  sessionId: null,
  textures: [],
};

const ui = {
  imageInput: document.getElementById("imageInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusText: document.getElementById("statusText"),
  previewImg: document.getElementById("previewImg"),
  resultImg: document.getElementById("resultImg"),
  texturesGrid: document.getElementById("texturesGrid"),
};

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
  ui.statusText.textContent = "Subiendo y analizando piso...";
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch("/api/analyze", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "No se pudo analizar la imagen");
  state.sessionId = data.session_id;
  ui.previewImg.src = `data:image/jpeg;base64,${data.mask_preview_base64}`;
  ui.statusText.textContent = data.message || "Analisis listo. Elige un material.";
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
