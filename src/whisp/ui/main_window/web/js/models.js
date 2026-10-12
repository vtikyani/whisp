// Speech models: one shared list for Settings and onboarding, kept fresh by bridge events.
import { useEffect } from "preact/hooks";
import { signal } from "@preact/signals";
import { html } from "./html.js";
import { call, on } from "./bridge.js";
import { formatBytes, languageName } from "./format.js";
import { Icon } from "./icons.js";
import { reportError, settings, toast } from "./store.js";
import { Badge, Button, IconButton, Pips, Progress, confirmDialog } from "./ui.js";

export const catalog = signal(null);
export const catalogError = signal(null);
export const downloads = signal({});
/** {done, total} while the GPU runtime pack downloads, else null. */
export const gpuPack = signal(null);

let inFlight = null;
let again = false;
let subscribed = false;

export function loadModels() {
  if (inFlight) {
    again = true;
    return inFlight;
  }
  inFlight = call("models.list")
    .then((result) => {
      catalog.value = result;
      catalogError.value = null;
      const live = {};
      for (const model of result.models) {
        if (model.downloading) live[model.id] = model.progress || downloads.value[model.id] || { done: 0, total: 0 };
      }
      downloads.value = live;
      if (result.gpu?.installing && !gpuPack.value) {
        gpuPack.value = { done: 0, total: result.gpu.runtime_download_size };
      } else if (!result.gpu?.installing) {
        gpuPack.value = null;
      }
    })
    .catch((error) => {
      catalogError.value = error;
    })
    .finally(() => {
      inFlight = null;
      if (again) {
        again = false;
        loadModels();
      }
    });
  return inFlight;
}

function subscribe() {
  if (subscribed) return;
  subscribed = true;
  on("models.progress", (payload) => {
    // Only for a download this page knows is running: an update for one that has ended would bring its bar back.
    if (!(payload.id in downloads.value)) return;
    downloads.value = { ...downloads.value, [payload.id]: { done: payload.done, total: payload.total } };
  });
  on("models.changed", () => loadModels());
  on("models.failed", (payload) => {
    const { [payload.id]: _failed, ...rest } = downloads.value;
    downloads.value = rest;
    toast(payload.message, { kind: "error" });
    loadModels();
  });
  on("settings.changed", (payload) => {
    if (catalog.value && payload.sections.includes("transcription")) loadModels();
  });
  on("gpu.progress", (payload) => {
    gpuPack.value = { done: payload.done, total: payload.total };
  });
  on("gpu.installed", (payload) => {
    gpuPack.value = null;
    if (payload.gpu.cuda_ready) toast("GPU speed is on. Dictation now runs on your graphics card.", { kind: "success" });
    else toast(payload.gpu.detail, { kind: "error" });
    loadModels();
  });
  on("gpu.failed", (payload) => {
    gpuPack.value = null;
    if (payload.message) toast(payload.message, { kind: "error" });
    loadModels();
  });
}

export async function installGpuPack(gpu) {
  gpuPack.value = { done: 0, total: gpu.runtime_download_size };
  try {
    await call("gpu.install_runtime");
  } catch (error) {
    gpuPack.value = null;
    reportError(error);
  }
}

export async function cancelGpuPack() {
  try {
    await call("gpu.cancel_runtime");
  } catch (error) {
    reportError(error);
  }
}

export function useModels() {
  useEffect(() => {
    subscribe();
    loadModels();
  }, []);
  return catalog.value;
}

export function variantOf(model, key = model.preferred_variant) {
  return model.variants.find((variant) => variant.key === key) || model.variants[0];
}

export async function downloadModel(model, variantKey = model.preferred_variant) {
  const variant = variantOf(model, variantKey);
  downloads.value = { ...downloads.value, [model.id]: { done: 0, total: variant.size } };
  try {
    await call("models.download", { id: model.id, variant: variant.key });
  } catch (error) {
    const { [model.id]: _started, ...rest } = downloads.value;
    downloads.value = rest;
    reportError(error);
  }
}

export async function cancelDownload(model) {
  const { [model.id]: _cancelled, ...rest } = downloads.value;
  downloads.value = rest;
  try {
    await call("models.cancel", { id: model.id });
  } catch (error) {
    reportError(error);
  }
  loadModels();
}

export async function deleteModel(model) {
  const ok = await confirmDialog({
    title: `Delete ${model.name}?`,
    text: "This frees up disk space. You can download it again any time.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  try {
    await call("models.delete", { id: model.id });
    toast(`${model.name} deleted.`);
  } catch (error) {
    reportError(error);
  }
  loadModels();
}

export async function selectModel(model) {
  try {
    settings.value = await call("models.select", { id: model.id });
    toast(`Whisp now uses ${model.name}.`, { kind: "success" });
  } catch (error) {
    reportError(error);
  }
}

export function languagesLabel(languages) {
  if (languages.length === 1) return languageName(languages[0]);
  return `${languages.length} languages`;
}

export function vramLabel(megabytes) {
  if (!megabytes) return "";
  return megabytes >= 1024 ? `${Math.round(megabytes / 1024)} GB` : `${megabytes} MB`;
}

export function gpuName(gpu) {
  return (gpu?.name || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
}

function DownloadRow({ model, progress }) {
  const total = progress.total || variantOf(model).size;
  const pct = total ? Math.min(100, Math.floor((progress.done / total) * 100)) : 0;
  return html`<div class="model-download">
    <${Progress} done=${progress.done} total=${total} indeterminate=${!progress.done} label=${`Downloading ${model.name}`} />
    <div class="model-download-meta">
      <span class="tnum">${progress.done ? `${pct}% · ${formatBytes(progress.done)} of ${formatBytes(total)}` : "Starting download…"}</span>
      <${Button} kind="ghost" size="sm" onClick=${() => cancelDownload(model)}>Cancel<//>
    </div>
  </div>`;
}

/** One speech model with its download / use / delete actions. */
export function ModelCard({ model, data, primary = false }) {
  const progress = downloads.value[model.id];
  const selected = data.selected === model.id;
  const variant = variantOf(model);
  const upgrade = model.installed && !variant.installed && !progress;
  // Describe what's on disk when the preferred version isn't; otherwise what a download would get.
  const shown = variant.installed || !model.installed ? variant : model.variants.find((v) => v.installed);
  const onGpu = shown.key === "gpu";

  let actions;
  if (progress) {
    actions = null;
  } else if (!model.installed) {
    actions = html`<${Button} kind=${primary ? "primary" : "secondary"} icon="download" onClick=${() => downloadModel(model)}>
      Download · ${formatBytes(variant.size)}
    <//>`;
  } else {
    actions = html`
      ${upgrade &&
      html`<${Button} kind="ghost" size="sm" icon="download" onClick=${() => downloadModel(model)}>
        Get ${variant.key === "gpu" ? "GPU" : "CPU"} version · ${formatBytes(variant.size)}
      <//>`}
      ${selected
        ? html`<${Badge} kind="success" icon="check">In use<//>`
        : html`<${Button} kind="secondary" size="sm" onClick=${() => selectModel(model)}>Use this model<//>`}
      <${IconButton}
        icon="trash-2"
        label=${selected ? "Choose another model before deleting this one" : "Delete model"}
        tipSide="left"
        danger
        disabled=${selected}
        onClick=${() => deleteModel(model)}
      />
    `;
  }

  return html`<article class=${`model${selected ? " is-selected" : ""}${primary ? " is-primary" : ""}`}>
    <div class="model-main">
      <div class="model-heading">
        <h3 class="model-name">${model.name}</h3>
        ${model.recommended && html`<${Badge} kind="accent">Recommended<//>`}
        ${model.installed && !selected && html`<${Badge}>Downloaded<//>`}
      </div>
      <p class="model-desc">${model.description}</p>
      <div class="model-meta">
        <span title=${model.languages.map(languageName).join(", ")}>
          <${Icon} name="languages" size=${13} />${languagesLabel(model.languages)}
        </span>
        <span><${Icon} name="hard-drive-download" size=${13} />${formatBytes(shown.size)}</span>
        <span><${Icon} name=${onGpu ? "microchip" : "cpu"} size=${13} />${onGpu ? "GPU" : "CPU"}</span>
        <a
          href=${model.license_url}
          onClick=${(event) => {
            event.preventDefault();
            call("app.open_url", { url: model.license_url }).catch(reportError);
          }}
          >${model.license}</a
        >
      </div>
    </div>
    <div class="model-side">
      <div class="model-scores">
        <span>Accuracy</span><${Pips} value=${model.accuracy} label="Accuracy" />
        <span>Speed</span><${Pips} value=${model.speed} label="Speed" />
      </div>
      ${actions && html`<div class="model-actions">${actions}</div>`}
    </div>
    ${progress && html`<${DownloadRow} model=${model} progress=${progress} />`}
  </article>`;
}

function GpuPackProgress({ progress }) {
  const { done, total } = progress;
  const pct = total ? Math.min(100, Math.floor((done / total) * 100)) : 0;
  return html`<div class="model-download">
    <${Progress} done=${done} total=${total} indeterminate=${!done} label="Downloading GPU support" />
    <div class="model-download-meta">
      <span class="tnum">${done ? `${pct}% · ${formatBytes(done)} of ${formatBytes(total)}` : "Starting download…"}</span>
      <${Button} kind="ghost" size="sm" onClick=${cancelGpuPack}>Cancel<//>
    </div>
  </div>`;
}

export function GpuCard({ gpu }) {
  if (!gpu) return null;
  const ready = gpu.cuda_ready;
  const name = gpu.has_nvidia ? gpu.name || "NVIDIA graphics card" : "No NVIDIA graphics card";
  const progress = gpuPack.value;
  const canInstall = !ready && gpu.has_nvidia && gpu.runtime_download_size > 0;
  let side;
  if (ready) {
    side = html`<${Badge} kind="success" icon="zap">GPU ready<//>`;
  } else if (canInstall && !progress) {
    side = html`<${Button} kind="secondary" size="sm" icon="zap" onClick=${() => installGpuPack(gpu)}>
      Turn on GPU speed · ${formatBytes(gpu.runtime_download_size)}
    <//>`;
  } else if (!progress) {
    side = html`<${Badge} icon="cpu">Runs on CPU<//>`;
  }
  return html`<div class=${`gpu${ready ? " is-ready" : ""}${progress ? " is-installing" : ""}`}>
    <div class="gpu-icon"><${Icon} name=${ready ? "microchip" : "cpu"} size=${18} /></div>
    <div class="gpu-text">
      <div class="gpu-name">${name}${gpu.vram_mb ? html`<span class="faint"> · ${vramLabel(gpu.vram_mb)}</span>` : ""}</div>
      <div class="gpu-detail">${gpu.detail}</div>
    </div>
    ${side}
    ${progress && html`<${GpuPackProgress} progress=${progress} />`}
  </div>`;
}
