const auth = {
  token: sessionStorage.getItem("adminToken") || "",
  user: null,
};

const ui = {
  loginForm: document.getElementById("loginForm"),
  dashboard: document.getElementById("dashboard"),
  loginBox: document.getElementById("loginBox"),
  loginEmail: document.getElementById("loginEmail"),
  loginPassword: document.getElementById("loginPassword"),
  loginStatus: document.getElementById("loginStatus"),
  meText: document.getElementById("meText"),
  texturesList: document.getElementById("texturesList"),
  usersList: document.getElementById("usersList"),
  textureForm: document.getElementById("textureForm"),
  userForm: document.getElementById("userForm"),
  aiForm: document.getElementById("aiForm"),
  aiTestForm: document.getElementById("aiTestForm"),
  aiPreview: document.getElementById("aiPreview"),
  globalStatus: document.getElementById("globalStatus"),
};

async function api(path, options = {}) {
  const headers = options.headers || {};
  if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Error de API");
  return data;
}

function showDashboard() {
  ui.loginBox.classList.add("hidden");
  ui.dashboard.classList.remove("hidden");
}

function showLogin() {
  ui.loginBox.classList.remove("hidden");
  ui.dashboard.classList.add("hidden");
}

async function loadMe() {
  const data = await api("/api/admin/me");
  auth.user = data.user;
  ui.meText.textContent = `${data.user.email} (${data.user.role})`;
}

async function loadTextures() {
  const data = await api("/api/admin/textures");
  const textures = data.textures || [];
  ui.texturesList.innerHTML = textures
    .map((t) => `<li class="border rounded p-2">${t.name} - ${t.category} - ${t.active ? "activo" : "inactivo"}</li>`)
    .join("");
}

async function loadUsers() {
  if (auth.user?.role !== "admin") {
    ui.usersList.innerHTML = `<li class="text-slate-500">Solo admin puede gestionar usuarios.</li>`;
    return;
  }
  const data = await api("/api/admin/users");
  ui.usersList.innerHTML = data.users
    .map((u) => `<li class="border rounded p-2">${u.email} - ${u.role} - ${u.active ? "activo" : "inactivo"}</li>`)
    .join("");
}

async function loadAiConfig() {
  if (auth.user?.role !== "admin") return;
  const cfg = await api("/api/admin/ai-config");
  for (const key of [
    "replicate_model",
    "floor_text_prompt",
    "floor_text_prompt_alt",
    "negative_mask_prompt",
    "environment_prompt",
    "furniture_subtraction_prompt",
    "mask_adjustment_factor",
    "detection_threshold",
    "box_threshold",
    "max_image_width",
    "mask_feather_px",
    "blend_strength",
  ]) {
    const el = document.getElementById(`ai_${key}`);
    if (el) el.value = cfg[key] ?? "";
  }
  document.getElementById("ai_enable_fallback_heuristic").checked = !!cfg.enable_fallback_heuristic;
  const envLayer = document.getElementById("ai_enable_environment_layer");
  if (envLayer) envLayer.checked = cfg.enable_environment_layer !== false;
  const furnSub = document.getElementById("ai_enable_furniture_subtraction");
  if (furnSub) furnSub.checked = cfg.enable_furniture_subtraction !== false;
  const colorRef = document.getElementById("ai_enable_color_refinement");
  if (colorRef) colorRef.checked = !!cfg.enable_color_refinement;
}

ui.loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const data = await api("/api/admin/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: ui.loginEmail.value, password: ui.loginPassword.value }),
    });
    auth.token = data.access_token;
    sessionStorage.setItem("adminToken", auth.token);
    showDashboard();
    await loadMe();
    await Promise.all([loadTextures(), loadUsers(), loadAiConfig()]);
  } catch (err) {
    ui.loginStatus.textContent = err.message;
  }
});

ui.textureForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const fd = new FormData(ui.textureForm);
    await api("/api/admin/textures", { method: "POST", body: fd });
    ui.textureForm.reset();
    await loadTextures();
    ui.globalStatus.textContent = "Textura subida correctamente.";
  } catch (err) {
    ui.globalStatus.textContent = err.message;
  }
});

ui.userForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: document.getElementById("user_email").value,
        password: document.getElementById("user_password").value,
        role: document.getElementById("user_role").value,
        active: true,
      }),
    });
    ui.userForm.reset();
    await loadUsers();
    ui.globalStatus.textContent = "Usuario creado.";
  } catch (err) {
    ui.globalStatus.textContent = err.message;
  }
});

ui.aiForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const payload = {
      replicate_model: document.getElementById("ai_replicate_model").value,
      floor_text_prompt: document.getElementById("ai_floor_text_prompt").value,
      floor_text_prompt_alt: document.getElementById("ai_floor_text_prompt_alt").value,
      negative_mask_prompt: document.getElementById("ai_negative_mask_prompt").value,
      environment_prompt: document.getElementById("ai_environment_prompt").value,
      furniture_subtraction_prompt: document.getElementById("ai_furniture_subtraction_prompt").value,
      enable_environment_layer: document.getElementById("ai_enable_environment_layer").checked,
      enable_furniture_subtraction: document.getElementById("ai_enable_furniture_subtraction").checked,
      enable_color_refinement: document.getElementById("ai_enable_color_refinement").checked,
      mask_adjustment_factor: Number(document.getElementById("ai_mask_adjustment_factor").value),
      detection_threshold: Number(document.getElementById("ai_detection_threshold").value),
      box_threshold: Number(document.getElementById("ai_box_threshold").value),
      max_image_width: Number(document.getElementById("ai_max_image_width").value),
      enable_fallback_heuristic: document.getElementById("ai_enable_fallback_heuristic").checked,
      mask_feather_px: Number(document.getElementById("ai_mask_feather_px").value),
      blend_strength: Number(document.getElementById("ai_blend_strength").value),
    };
    await api("/api/admin/ai-config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    ui.globalStatus.textContent = "Configuracion IA guardada.";
  } catch (err) {
    ui.globalStatus.textContent = err.message;
  }
});

const aiTestPreviews = { env: "", raw: "", floor: "" };

function showAiTestTab(tab) {
  const src = aiTestPreviews[tab];
  if (src) document.getElementById("aiPreview").src = src;
  document.querySelectorAll("[data-ai-tab]").forEach((btn) => {
    btn.classList.toggle("border-sky-600", btn.dataset.aiTab === tab);
  });
}

document.querySelectorAll("[data-ai-tab]").forEach((btn) => {
  btn.addEventListener("click", () => showAiTestTab(btn.dataset.aiTab));
});

ui.aiTestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const fd = new FormData(ui.aiTestForm);
    const data = await api("/api/admin/ai-config/test", { method: "POST", body: fd });
    const b64 = (s) => (s ? `data:image/jpeg;base64,${s}` : "");
    aiTestPreviews.floor = b64(data.preview_base64);
    aiTestPreviews.env = b64(data.environment_preview_base64);
    aiTestPreviews.raw = b64(data.raw_floor_preview_base64) || aiTestPreviews.floor;
    document.getElementById("aiPreviewTabs").classList.remove("hidden");
    showAiTestTab("floor");
    ui.globalStatus.textContent = data.message || "Prueba completada.";
  } catch (err) {
    ui.globalStatus.textContent = err.message;
  }
});

async function init() {
  if (!auth.token) {
    showLogin();
    return;
  }
  try {
    showDashboard();
    await loadMe();
    await Promise.all([loadTextures(), loadUsers(), loadAiConfig()]);
  } catch {
    sessionStorage.removeItem("adminToken");
    auth.token = "";
    showLogin();
  }
}

init();
