const $ = (id) => document.getElementById(id);
let currentJobId = null,
  pollTimer = null,
  previewTimer = null,
  editTimer = null;
let activeRunning = false,
  geometryReady = false,
  inspectionRevision = 0;
let preview = null,
  previewPending = false,
  previewRevision = 0,
  previewView = "raw";
let references = new Set(),
  resultFilter = "all",
  sessionJobs = [];
const tileKey = (row, column) => `${row},${column}`;
const escapeHtml = (text) => {
  const element = document.createElement("span");
  element.textContent = String(text ?? "");
  return element.innerHTML;
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false)
    throw new Error(payload.error || "Request failed.");
  return payload;
}
function post(path, payload) {
  return api(path, { method: "POST", body: JSON.stringify(payload) });
}
function formError(message = "") {
  $("formError").hidden = !message;
  $("formError").textContent = message;
}
function formatBytes(value) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (value >= 1024 && i < 3) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}
function updateRunButton() {
  $("runButton").disabled =
    activeRunning ||
    previewPending ||
    !geometryReady ||
    (preview && references.size < 2);
  $("previewButton").disabled =
    activeRunning || previewPending || !geometryReady;
}
function populateImages(images) {
  $("imageSelect").replaceChildren();
  if (!images.length) {
    $("imageSelect").add(
      new Option("No TIFFs found — choose an image folder", ""),
    );
    $("imageSelect").disabled = true;
    geometryReady = false;
    invalidatePreview();
    $("geometryCard").textContent =
      "No TIFF volumes were found in this folder.";
    updateRunButton();
    return;
  }
  $("imageSelect").disabled = false;
  for (const image of images) {
    const option = new Option(
      `${image.relative_path} · ${formatBytes(image.size_bytes)}`,
      image.path,
    );
    option.dataset.fov = image.suggested_fov_um;
    option.dataset.reason = image.suggestion_reason;
    $("imageSelect").add(option);
  }
  imageChanged();
}
async function loadConfig() {
  try {
    const config = await api("/api/config");
    $("inputFolder").value = config.input_dir;
    $("outputFolder").value = config.output_dir;
    populateImages(config.images);
    await refreshHistory(true);
  } catch (error) {
    formError(
      error.message +
        " Start the app with the operating system launcher, rather than opening index.html directly.",
    );
  }
}
async function scanFolder() {
  formError();
  $("scanButton").disabled = true;
  try {
    const data = await post("/api/scan", { folder: $("inputFolder").value });
    $("inputFolder").value = data.input_dir;
    populateImages(data.images);
  } catch (error) {
    formError(error.message);
  } finally {
    $("scanButton").disabled = false;
  }
}
async function browseLocal(kind) {
  formError();
  const buttons = [$("browseImageButton"), $("browseFolderButton")];
  buttons.forEach((button) => { button.disabled = true; });
  const button = kind === "image" ? buttons[0] : buttons[1];
  const label = button.textContent;
  button.textContent = "Choose in system dialog…";
  try {
    const data = await post("/api/browse", {kind, initial_dir: $("inputFolder").value});
    if (!data.cancelled) {
      $("inputFolder").value = data.input_dir;
      populateImages(data.images);
    }
  } catch (error) {
    formError(error.message);
  } finally {
    button.textContent = label;
    buttons.forEach((item) => { item.disabled = false; });
  }
}
function imageChanged() {
  const option = $("imageSelect").selectedOptions[0];
  if (!option?.value) return;
  $("inputLayout").value = "metadata";
  for (const id of ["voxelZ", "voxelY", "voxelX"]) $(id).value = "";
  $("fovInput").value = option.dataset.fov || 700;
  $("fovHint").textContent =
    `Autofilled: ${option.dataset.reason}. You can type a different patch FOV.`;
  invalidatePreview();
  inspectImage();
}
function invalidatePreview() {
  previewRevision++;
  clearInterval(previewTimer);
  previewTimer = null;
  preview = null;
  references.clear();
  previewPending = false;
  $("previewContent").hidden = true;
  $("previewLoading").hidden = true;
  $("previewError").hidden = true;
  $("previewEmpty").hidden = false;
  $("runNote").textContent =
    "Use automatic references, or review and edit them in the mosaic.";
  updateRunButton();
}
function geometryPayload() {
  return {
    image_path: $("imageSelect").value,
    fov_um: Number($("fovInput").value),
    auto_mode: $("autoMode").checked,
    rows: $("rowsInput").value || null,
    columns: $("columnsInput").value || null,
    input_layout: $("inputLayout").value,
    voxel_size_um_zyx: [$("voxelZ").value, $("voxelY").value, $("voxelX").value],
  };
}
async function inspectImage() {
  const revision = ++inspectionRevision;
  geometryReady = false;
  updateRunButton();
  if (!$("imageSelect").value) return;
  formError();
  $("geometryCard").innerHTML =
    '<span class="geometry-placeholder">Reading TIFF calibration…</span>';
  try {
    const data = await post("/api/inspect", geometryPayload());
    if (revision !== inspectionRevision) return;
    const g = data.geometry;
    $("geometryCard").innerHTML =
      `<div class="geometry-grid"><div><small>Volume · Z × Y × X</small><strong>${g.shape_zyx.join(" × ")}</strong></div><div><small>Voxel · ${escapeHtml(data.input_provenance?.calibration_source || "TIFF metadata")}</small><strong>${g.voxel_size_um_zyx.map((v) => Number(v).toFixed(3)).join(" × ")} µm</strong></div><div><small>Patch grid · rows × cols</small><strong>${g.rows} × ${g.columns}</strong></div><div><small>Patch · Y × X</small><strong>${g.patch_shape_yx.join(" × ")} px</strong></div></div>`;
    geometryReady = true;
  } catch (error) {
    $("calibrationDetails").open = true;
    if (revision === inspectionRevision)
      $("geometryCard").innerHTML =
        `<div class="form-error">${escapeHtml(error.message)}</div>`;
  } finally {
    if (revision === inspectionRevision) updateRunButton();
  }
}
function geometryEdited() {
  clearTimeout(editTimer);
  invalidatePreview();
  geometryReady = false;
  updateRunButton();
  editTimer = setTimeout(inspectImage, 400);
}
async function generatePreview() {
  formError();
  invalidatePreview();
  const revision = previewRevision;
  previewPending = true;
  updateRunButton();
  $("previewEmpty").hidden = true;
  $("previewLoading").hidden = false;
  $("previewProgress").textContent = "Sampling the full Z stack…";
  try {
    const data = await post("/api/previews", geometryPayload());
    if (revision !== previewRevision) return;
    previewTimer = setInterval(async () => {
      try {
        const payload = await api(`/api/previews/${data.preview.id}`);
        if (revision !== previewRevision) return;
        const task = payload.preview;
        $("previewProgress").textContent =
          `Preparing initial mosaic · ${task.progress}%`;
        if (task.status === "completed") {
          clearInterval(previewTimer);
          previewTimer = null;
          preview = task;
          previewPending = false;
          references = new Set(task.result.selected.map((p) => tileKey(...p)));
          previewView = "raw";
          renderPreview();
          updateRunButton();
        } else if (task.status === "failed") {
          clearInterval(previewTimer);
          previewTimer = null;
          previewFailure(task.error);
        }
      } catch (error) {
        if (revision === previewRevision) previewFailure(error.message);
      }
    }, 750);
  } catch (error) {
    if (revision === previewRevision) previewFailure(error.message);
  }
}
function previewFailure(message) {
  clearInterval(previewTimer);
  previewTimer = null;
  previewPending = false;
  $("previewLoading").hidden = true;
  $("previewError").hidden = false;
  $("previewError").textContent = message;
  $("previewEmpty").hidden = false;
  updateRunButton();
}
function renderPreview() {
  $("previewLoading").hidden = true;
  $("previewEmpty").hidden = true;
  $("previewContent").hidden = false;
  const { geometry, patches, preview_shape_yx } = preview.result;
  const ratio = preview_shape_yx[1] / preview_shape_yx[0];
  $("mosaicStage").style.aspectRatio = String(ratio);
  $("mosaicStage").style.maxWidth = `${Math.min(900, 620 * ratio)}px`;
  $("tileGrid").style.gridTemplateRows = `repeat(${geometry.rows},1fr)`;
  $("tileGrid").style.gridTemplateColumns = `repeat(${geometry.columns},1fr)`;
  $("tileGrid").replaceChildren();
  for (let row = 0; row < geometry.rows; row++)
    for (let column = 0; column < geometry.columns; column++) {
      const key = tileKey(row, column);
      const report = patches.find((p) => p.row === row && p.column === column);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tile-button";
      button.dataset.key = key;
      button.title = `Tile (${key}) · blank score ${report?.raw_background_score.toFixed(2)} · click to toggle reference`;
      button.innerHTML = `<span>${key}</span>`;
      button.addEventListener("click", () => {
        references.has(key) ? references.delete(key) : references.add(key);
        updateReferences();
      });
      $("tileGrid").append(button);
    }
  $("tileReportBody").innerHTML = patches
    .map(
      (p) =>
        `<tr><td>${p.row},${p.column}</td><td>${p.raw_background_score.toFixed(2)}</td><td>Provisional · raw tile statistics</td></tr>`,
    )
    .join("");
  $("previewDisclaimer").textContent =
    "The initial projection includes every Z slice at sampled XY positions. Automatic references are candidates, not confirmed empty tiles. Tissue contrast subtracts each tile’s median depth profile for display only. Select at least two tiles with no tissue.";
  showPreviewView("raw");
  updateReferences();
}
function showPreviewView(view) {
  if (!preview) return;
  previewView = view;
  $("mosaicImage").src = `/api/previews/${preview.id}/image?view=${view}`;
  $("rawViewButton").setAttribute("aria-pressed", String(view === "raw"));
  $("contrastViewButton").setAttribute(
    "aria-pressed",
    String(view === "contrast"),
  );
}
function updateReferences() {
  for (const button of $("tileGrid").children) {
    const selected = references.has(button.dataset.key);
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute(
      "aria-label",
      `Tile ${button.dataset.key}; ${selected ? "included as blank reference" : "not a reference"}`,
    );
  }
  $("referenceCount").textContent = `${references.size} reference tiles`;
  $("runNote").textContent =
    references.size >= 2
      ? `Your ${references.size} reviewed reference tiles will be used for correction.`
      : "Select at least two reference tiles to run correction.";
  updateRunButton();
}
function collectForm() {
  const form = {
    ...geometryPayload(),
    action: document.querySelector('input[name="action"]:checked').value,
    background_patches: $("backgroundInput").value,
    strength: Number($("strengthInput").value),
    interface_margin_um: Number($("interfaceMarginInput").value),
    top_k: Number($("topKInput").value),
    noise_sigma: Number($("noiseInput").value),
    block_size_um: Number($("blockInput").value),
    minimum_separation_um: Number($("minimumSeparationInput").value),
    refractive_index: $("refractiveInput").value || null,
    output_dir: $("outputFolder").value,
  };
  if (preview) {
    form.reference_patches = [...references].map((key) =>
      key.split(",").map(Number),
    );
    form.reference_preview_id = preview.id;
  }
  return form;
}
async function runAnalysis() {
  formError();
  const config = collectForm();
  if (!config.image_path) return formError("Select a TIFF volume.");
  if (!Number.isFinite(config.fov_um) || config.fov_um <= 0)
    return formError("Enter a positive patch FOV.");
  if (!config.output_dir.trim())
    return formError("Enter an output folder in Advanced parameters.");
  activeRunning = true;
  updateRunButton();
  try {
    const data = await post("/api/jobs", config);
    currentJobId = data.job.id;
    $("resultsSection").hidden = true;
    renderJob(data.job);
    startJobPolling();
  } catch (error) {
    activeRunning = false;
    updateRunButton();
    formError(error.message);
  }
}
function startJobPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 1000);
}
async function pollJob() {
  if (!currentJobId) return;
  try {
    const data = await api(`/api/jobs/${currentJobId}`);
    $("connectionBanner").hidden = true;
    renderJob(data.job);
    if (["completed", "failed", "cancelled"].includes(data.job.status)) {
      clearInterval(pollTimer);
      pollTimer = null;
      activeRunning = false;
      updateRunButton();
      if (data.job.status === "completed") renderResults(data.job);
      await refreshHistory();
    }
  } catch (error) {
    $("connectionBanner").hidden = false;
    $("connectionBanner").textContent =
      "Connection to the local server was lost. Waiting to reconnect; do not start another run.";
  }
}
function renderJob(job) {
  $("emptyMonitor").hidden = true;
  $("activeMonitor").hidden = false;
  const errors = job.result?.errors || [];
  const withErrors = job.status === "completed" && errors.length;
  const status = withErrors ? "completed-with-errors" : job.status;
  $("statusBadge").className = `status-badge ${status}`;
  $("statusBadge").textContent = withErrors
    ? "Completed with detection error"
    : job.status[0].toUpperCase() + job.status.slice(1);
  $("stageText").textContent = job.stage;
  $("progressText").textContent = `${job.progress}%`;
  $("progressBar").style.width = `${job.progress}%`;
  $("runId").textContent = `RUN ${job.id}`;
  $("runOutput").textContent = job.output_dir;
  $("logOutput").textContent = (job.logs || []).join("\n");
  $("logOutput").scrollTop = $("logOutput").scrollHeight;
  const elapsed = Math.max(
    0,
    Math.round(
      (new Date(job.finished_at || Date.now()) - new Date(job.created_at)) /
        1000,
    ),
  );
  $("elapsedTime").textContent =
    elapsed >= 60
      ? `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`
      : `${elapsed}s`;
  const bounds = [25, 48, 70, 94, 100];
  [...$("pipelineSteps").children].forEach((li, index) =>
    li.classList.toggle("done", job.progress >= bounds[index]),
  );
  activeRunning = ["queued", "running", "cancelling"].includes(job.status);
  $("cancelButton").hidden = !activeRunning;
  const hasError = withErrors || ["failed", "cancelled"].includes(job.status);
  $("jobError").hidden = !hasError;
  if (hasError) {
    $("jobErrorTitle").textContent = withErrors
      ? "Glass pair detection failed; preprocessing completed"
      : "Analysis stopped";
    $("jobErrorText").textContent = withErrors
      ? errors.map((e) => e.message).join(" ")
      : job.error;
    $("technicalError").textContent = withErrors
      ? errors.map((e) => `${e.code}: ${e.technical_detail}`).join("\n")
      : job.technical_error || "See the processing log.";
  }
  updateRunButton();
}
function metric(label, value) {
  return `<div class="summary-metric"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`;
}
function renderResults(job) {
  const result = job.result || {},
    geometry = result.geometry || {};
  $("resultImageName").textContent =
    job.image_name || job.output_dir.split(/[\\/]/).pop();
  let metrics =
    metric("Patch grid", `${geometry.rows} × ${geometry.columns}`) +
    metric("Patch FOV", `${geometry.patch_fov_um} µm`);
  if (result.preprocessing)
    metrics += metric(
      "Tissue footprint",
      `${(result.preprocessing.mask_fraction * 100).toFixed(1)}%`,
    );
  if (result.patch_sandwich)
    metrics += metric(
      "Median glass gap",
      `${result.patch_sandwich.median_sandwich_depth_um.toFixed(2)} µm`,
    );
  else if (result.coverslip_spacing)
    metrics += metric(
      "Median glass gap",
      `${result.coverslip_spacing.central_field.optical_path_separation_um.median.toFixed(2)} µm`,
    );
  else
    metrics += metric(
      "Blank references",
      String((result.background_patches || []).length),
    );
  $("summaryStrip").innerHTML = metrics;
  $("resultGrid").replaceChildren();
  for (const file of result.files || []) {
    const url = `/api/download?job=${encodeURIComponent(job.id)}&file=${encodeURIComponent(file.path)}`;
    const card = document.createElement("article");
    card.className = "result-card";
    card.dataset.kind = file.kind;
    if (file.kind === "image") {
      card.innerHTML = `<button class="result-preview" aria-label="Enlarge ${escapeHtml(file.label)}"><img src="${url}&inline=1" alt="${escapeHtml(file.label)}" loading="lazy"></button><div class="result-info"><h3>${escapeHtml(file.label)}</h3><a href="${url}" download>Download ↓</a></div>`;
      card.querySelector("button").addEventListener("click", () => {
        $("imageDialogTitle").textContent = file.label;
        $("imageDialogImage").src = url + "&inline=1";
        $("imageDialog").showModal();
      });
    } else
      card.innerHTML = `<div class="download-card"><span class="download-symbol" aria-hidden="true">↓</span><h3>${escapeHtml(file.label)}</h3><a href="${url}" download>Download</a></div>`;
    $("resultGrid").append(card);
  }
  $("openFolderLink").dataset.jobId = job.id;
  $("openFolderLink").title = job.output_dir;
  $("resultsSection").hidden = false;
  filterResults(resultFilter);
}
function filterResults(filter) {
  resultFilter = filter;
  for (const card of $("resultGrid").children)
    card.hidden = filter !== "all" && card.dataset.kind !== filter;
  [
    ["showAllResults", "all"],
    ["showImages", "image"],
    ["showDownloads", "download"],
  ].forEach(([id, value]) =>
    $(id).setAttribute("aria-pressed", String(filter === value)),
  );
}
async function refreshHistory(restore = false) {
  const data = await api("/api/jobs");
  sessionJobs = data.jobs;
  $("historySection").hidden = !sessionJobs.length;
  $("historyList").replaceChildren();
  for (const job of sessionJobs) {
    const row = document.createElement("div");
    row.className = "history-row";
    row.innerHTML = `<b>${escapeHtml(job.image_name || job.id)}</b><span>${escapeHtml(job.status)} · ${escapeHtml(job.created_at.replace("T", " "))}</span><button class="text-button">Inspect run ↗</button>`;
    row.querySelector("button").addEventListener("click", async () => {
      const data = await api(`/api/jobs/${job.id}`);
      currentJobId = job.id;
      renderJob(data.job);
      if (data.job.status === "completed") {
        renderResults(data.job);
        $("resultsSection").scrollIntoView({ behavior: "smooth" });
      }
      if (activeRunning) startJobPolling();
    });
    $("historyList").append(row);
  }
  if (restore) {
    const active = sessionJobs.find((job) =>
      ["queued", "running", "cancelling"].includes(job.status),
    );
    if (active) {
      currentJobId = active.id;
      renderJob(active);
      startJobPolling();
    }
  }
}
async function cancelRun() {
  try {
    if (currentJobId) await post(`/api/jobs/${currentJobId}/cancel`, {});
  } catch (error) {
    formError(error.message);
  }
}
async function quit() {
  $("quitConfirm").disabled = true;
  try {
    await post("/api/quit", {});
    clearInterval(pollTimer);
    clearInterval(previewTimer);
    document.body.innerHTML =
      '<main style="min-height:100vh;display:grid;place-content:center;text-align:center"><img src="/favicon.svg" alt="" style="width:64px;margin:auto"><h1>Q3Slide is closed.</h1><p>Your saved results remain in their output folders. You can close this tab.</p></main>';
  } catch (error) {
    $("quitConfirm").disabled = false;
    formError(error.message);
  }
}
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("q3slide-theme", theme);
  } catch {}
}
try {
  setTheme(localStorage.getItem("q3slide-theme") || "light");
} catch {}
$("themeButton").addEventListener("click", () =>
  setTheme(
    document.documentElement.dataset.theme === "light" ? "dark" : "light",
  ),
);
$("scanButton").addEventListener("click", scanFolder);
$("browseImageButton").addEventListener("click", () => browseLocal("image"));
$("browseFolderButton").addEventListener("click", () => browseLocal("folder"));
$("imageSelect").addEventListener("change", imageChanged);
$("inputLayout").addEventListener("change", geometryEdited);
for (const id of ["voxelZ", "voxelY", "voxelX"]) $(id).addEventListener("input", geometryEdited);
$("fovInput").addEventListener("input", geometryEdited);
$("rowsInput").addEventListener("input", geometryEdited);
$("columnsInput").addEventListener("input", geometryEdited);
$("autoMode").addEventListener("change", () => {
  $("rowsInput").disabled = $("autoMode").checked;
  $("columnsInput").disabled = $("autoMode").checked;
  $("backgroundInput").disabled = $("autoMode").checked;
  geometryEdited();
});
$("advancedToggle").addEventListener("click", () => {
  const panel = $("advancedPanel");
  panel.hidden = !panel.hidden;
  $("advancedToggle").setAttribute("aria-expanded", String(!panel.hidden));
  $("advancedToggle").querySelector("span").textContent = panel.hidden
    ? "+"
    : "−";
});
$("previewButton").addEventListener("click", generatePreview);
$("rawViewButton").addEventListener("click", () => showPreviewView("raw"));
$("contrastViewButton").addEventListener("click", () =>
  showPreviewView("contrast"),
);
$("restoreReferences").addEventListener("click", () => {
  references = new Set(preview.result.selected.map((p) => tileKey(...p)));
  updateReferences();
});
$("clearReferences").addEventListener("click", () => {
  references.clear();
  updateReferences();
});
$("runButton").addEventListener("click", runAnalysis);
$("cancelButton").addEventListener("click", cancelRun);
$("copyLog").addEventListener("click", async (event) => {
  event.preventDefault();
  try {
    await navigator.clipboard.writeText($("logOutput").textContent);
    $("copyLog").textContent = "Copied";
    setTimeout(() => ($("copyLog").textContent = "Copy"), 1500);
  } catch {
    formError("Select the log text to copy it.");
  }
});
$("openFolderLink").addEventListener("click", async () => {
  try {
    await post("/api/open-folder", {
      job_id: $("openFolderLink").dataset.jobId,
    });
  } catch (error) {
    formError(error.message);
  }
});
$("showAllResults").addEventListener("click", () => filterResults("all"));
$("showImages").addEventListener("click", () => filterResults("image"));
$("showDownloads").addEventListener("click", () => filterResults("download"));
$("quitButton").addEventListener("click", () => $("quitDialog").showModal());
$("quitCancel").addEventListener("click", () => $("quitDialog").close());
$("quitConfirm").addEventListener("click", quit);
$("methodButton").addEventListener("click", () =>
  $("methodDialog").showModal(),
);
$("methodClose").addEventListener("click", () => $("methodDialog").close());
$("imageClose").addEventListener("click", () => $("imageDialog").close());
$("rowsInput").disabled = true;
$("columnsInput").disabled = true;
$("backgroundInput").disabled = true;
updateRunButton();
loadConfig();
