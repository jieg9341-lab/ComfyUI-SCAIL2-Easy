import { app } from "../../scripts/app.js";

const EXT_NAME = "Comfy.SCAIL2.Easy";
const SIMPLE_CHUNK_ADVANCED_WIDGETS = ["max_frames", "chunk_frames", "overlap_frames", "color_correction"];
const SIMPLE_CONTEXT_ADVANCED_WIDGETS = ["max_frames", "context_frames", "context_overlap_frames"];
const SIMPLE_ADVANCED_MODE_WIDGET = "long_video_mode";
const SIMPLE_WIDGET_ORDER = [
  "advanced",
  "long_video_mode",
  "max_frames",
  "chunk_frames",
  "overlap_frames",
  "color_correction",
  "context_frames",
  "context_overlap_frames",
];
const SIMPLE_ALL_ADVANCED_WIDGETS = Array.from(new Set([SIMPLE_ADVANCED_MODE_WIDGET, ...SIMPLE_CHUNK_ADVANCED_WIDGETS, ...SIMPLE_CONTEXT_ADVANCED_WIDGETS]));
const MAX_REFERENCE_SUBJECTS = 6;
const MAX_MIXED_REFERENCE_IMAGES = 5;
const SIMPLE_COMBO_LABELS = {
  zh: {
    mode: {
      replacement: "\u89d2\u8272\u66ff\u6362",
      animation: "\u52a8\u4f5c\u8fc1\u79fb",
    },
    long_video_mode: {
      chunk: "\u63a5\u7eed\u5206\u6bb5",
      context_sampling: "\u4e0a\u4e0b\u6587\u91c7\u6837",
    },
  },
  en: {
    mode: {
      replacement: "Replacement",
      animation: "Animation",
    },
    long_video_mode: {
      chunk: "Chunk",
      context_sampling: "Context sampling",
    },
  },
};
const WIDGET_DEFAULTS = new WeakMap();
const FIT_VIDEO_SETUP = new WeakSet();
const REFERENCE_PACK_SETUP = new WeakSet();
const SIMPLE_VIDEO_SETUP = new WeakSet();
const hiddenWidgetComputeSize = () => [0, -4];
const EASY_TRANSLATIONS = {
  zh: {
    SCAIL2SimpleVideo: {
      inputs: {
        reference_image: "\u53c2\u8003\u56fe",
        advanced: "\u9ad8\u7ea7\u53c2\u6570",
        long_video_mode: "\u957f\u89c6\u9891\u65b9\u5f0f",
        max_frames: "\u6700\u5927\u5e27\u6570",
        chunk_frames: "\u6bcf\u6bb5\u5e27\u6570",
        overlap_frames: "\u91cd\u53e0\u5e27\u6570",
        color_correction: "\u8272\u5f69\u6821\u6b63",
        context_frames: "\u4e0a\u4e0b\u6587\u7a97\u53e3\u5e27\u6570",
        context_overlap_frames: "\u4e0a\u4e0b\u6587\u91cd\u53e0\u5e27\u6570",
      },
    },
    SCAIL2ReferencePack: {
      inputs: {
        subject_count: "\u4e3b\u4f53\u6570\u91cf",
        reference_count: "\u53c2\u8003\u56fe\u6570\u91cf",
        scene_image: "\u80cc\u666f\u56fe",
      },
    },
    SCAIL2ReferenceSAMBuilder: {
      inputs: {
        reference_pack: "\u53c2\u8003\u5305",
        sam_model: "SAM\u6a21\u578b",
        conditioning: "SAM\u6761\u4ef6",
        detection_threshold: "\u68c0\u6d4b\u9608\u503c",
        max_objects: "\u6700\u5927\u5bf9\u8c61\u6570",
        detect_interval: "\u68c0\u6d4b\u95f4\u9694",
      },
    },
  },
  en: {
    SCAIL2SimpleVideo: {
      inputs: {
        reference_image: "Reference image",
        advanced: "Advanced",
        long_video_mode: "Long video mode",
        max_frames: "Max frames",
        chunk_frames: "Chunk frames",
        overlap_frames: "Overlap frames",
        color_correction: "Color correction",
        context_frames: "Context frames",
        context_overlap_frames: "Context overlap",
      },
    },
    SCAIL2ReferencePack: {
      inputs: {
        subject_count: "Subjects",
        reference_count: "References",
        scene_image: "Background image",
      },
    },
    SCAIL2ReferenceSAMBuilder: {
      inputs: {
        reference_pack: "Reference pack",
        sam_model: "SAM model",
        conditioning: "Conditioning",
        detection_threshold: "Detection threshold",
        max_objects: "Max objects",
        detect_interval: "Detect interval",
      },
    },
  },
};

for (let subject = 1; subject <= MAX_REFERENCE_SUBJECTS; subject++) {
  EASY_TRANSLATIONS.zh.SCAIL2ReferencePack.inputs[`subject_${subject}_image`] = `\u4e3b\u4f53${subject}`;
  EASY_TRANSLATIONS.en.SCAIL2ReferencePack.inputs[`subject_${subject}_image`] = `Subject ${subject}`;
  EASY_TRANSLATIONS.zh.SCAIL2ReferencePack.inputs[`subject_${subject}_image_1`] = `\u4e3b\u4f53${subject}`;
  EASY_TRANSLATIONS.en.SCAIL2ReferencePack.inputs[`subject_${subject}_image_1`] = `Subject ${subject}`;
}
for (let reference = 1; reference <= MAX_MIXED_REFERENCE_IMAGES; reference++) {
  EASY_TRANSLATIONS.zh.SCAIL2ReferencePack.inputs[`reference_${reference}`] = `\u53c2\u8003\u56fe${reference}`;
  EASY_TRANSLATIONS.en.SCAIL2ReferencePack.inputs[`reference_${reference}`] = `Reference ${reference}`;
}

function isChineseLocale() {
  const htmlLang = String(document.documentElement?.lang || "").toLowerCase();
  if (htmlLang.startsWith("zh")) return true;
  return (navigator.languages || [navigator.language || ""]).some((lang) => String(lang).toLowerCase().startsWith("zh"));
}

function classNameForNode(node) {
  return node?.constructor?.comfyClass || node?.constructor?.type || node?.type;
}

function applyLabel(item, label) {
  if (!item || !label) return;
  item.label = label;
  item.localized_name = label;
  item.display_name = label;
  if (item.options) {
    item.options.label = label;
    item.options.localized_name = label;
    item.options.display_name = label;
  }
  if (item._state) {
    item._state.label = label;
    item._state.localized_name = label;
    item._state.display_name = label;
    item._state.options ||= {};
    item._state.options.label = label;
    item._state.options.localized_name = label;
    item._state.options.display_name = label;
  }
}

function translationForInput(className, name) {
  const locale = isChineseLocale() ? "zh" : "en";
  return EASY_TRANSLATIONS[locale]?.[className]?.inputs?.[name];
}

function localizeComboWidget(widget, name) {
  if (!widget) return;
  ensureOptions(widget);
  widget.options.getOptionLabel = (value) => {
    const raw = value == null ? "" : String(value);
    const locale = isChineseLocale() ? "zh" : "en";
    return SIMPLE_COMBO_LABELS[locale]?.[name]?.[raw] || raw;
  };
  if (widget._state) {
    widget._state.options ||= {};
    widget._state.options.getOptionLabel = widget.options.getOptionLabel;
  }
}

function applyEasyNodeDataTranslation(nodeData) {
  const locale = isChineseLocale() ? "zh" : "en";
  const translation = EASY_TRANSLATIONS[locale]?.[nodeData?.name];
  if (!translation) return;
  const translateInputs = (inputs) => {
    if (!inputs) return;
    for (const [name, config] of Object.entries(inputs)) {
      const label = translation.inputs?.[name];
      if (!label || !Array.isArray(config)) continue;
      if (!config[1] || typeof config[1] !== "object" || Array.isArray(config[1])) {
        config[1] = {};
      }
      config[1].label = label;
      config[1].localized_name = label;
      config[1].display_name = label;
    }
  };
  translateInputs(nodeData.input?.required);
  translateInputs(nodeData.input?.optional);
}

function trimReferencePackNodeDataInputs(nodeData) {
  const optional = nodeData?.input?.optional;
  if (optional && !Array.isArray(optional)) {
    for (const name of Object.keys(optional)) {
      if (isReferencePackCountWidgetName(name)) continue;
      if (!isReferencePackManagedInputName(name)) continue;
      if (name !== "subject_1_image" && name !== "scene_image") {
        delete optional[name];
      }
    }
  }

  if (Array.isArray(nodeData?.input_order?.required)) {
    nodeData.input_order.required = nodeData.input_order.required.filter((name) => !isReferencePackManagedInputName(name));
  }
  if (Array.isArray(nodeData?.input_order?.optional)) {
    nodeData.input_order.optional = nodeData.input_order.optional.filter((name) => (
      isReferencePackCountWidgetName(name)
      || (!isReferencePackManagedInputName(name) || name === "subject_1_image" || name === "scene_image")
    ));
  }
}

function applyEasyNodeInstanceTranslation(node) {
  const className = classNameForNode(node);
  for (const input of node.inputs || []) {
    applyLabel(input, translationForInput(className, input.name));
  }
  for (const widget of node.widgets || []) {
    applyLabel(widget, translationForInput(className, widget.name));
  }
}

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function clampInt(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function isReferencePackManagedInputName(name) {
  const text = String(name || "");
  return name === "scene_image"
    || /^subject_\d+_image$/.test(text)
    || /^reference_\d+$/.test(text)
    || /^subject_\d+_image_\d+$/.test(text);
}

function isReferencePackCountWidgetName(name) {
  return name === "subject_count" || name === "reference_count";
}

function isLegacyReferencePackRefCountWidgetName(name) {
  return /^subject_\d+_ref_count$/.test(String(name || ""));
}

function clampReferencePackWidgetValue(widget, name, value) {
  const min = name === "reference_count" ? 0 : 1;
  const max = name === "subject_count" ? MAX_REFERENCE_SUBJECTS : MAX_MIXED_REFERENCE_IMAGES;
  const source = value ?? widget?.value ?? 1;
  const clamped = clampInt(source, min, max);
  if (widget) widget.value = clamped;
  return clamped;
}

function makeReferencePackInput(name) {
  const input = {
    name,
    type: "IMAGE",
    link: null,
  };
  applyLabel(input, translationForInput("SCAIL2ReferencePack", name));
  return input;
}

function findInputByName(node, name) {
  return (node.inputs || []).find((input) => input?.name === name);
}

function removeInputAt(node, index) {
  if (typeof node.removeInput === "function") {
    node.removeInput(index);
    return;
  }
  if (node.inputs?.[index]?.link != null) {
    node.disconnectInput?.(index);
  }
  node.inputs?.splice(index, 1);
}

function updateInputLinkTargets(node) {
  const graph = node?.graph || app.graph;
  if (!graph || !Array.isArray(node?.inputs)) return;
  node.inputs.forEach((input, index) => {
    if (input?.link == null) return;
    const link = graph.links?.[input.link];
    if (link) link.target_slot = index;
  });
}

function removeOutputAt(node, index) {
  if (typeof node.removeOutput === "function") {
    node.removeOutput(index);
    return;
  }
  const output = node.outputs?.[index];
  if (Array.isArray(output?.links)) {
    for (const linkId of [...output.links]) {
      node.disconnectOutput?.(index, linkId);
    }
  }
  node.outputs?.splice(index, 1);
}

function updateOutputLinkOrigins(node) {
  const graph = node?.graph || app.graph;
  if (!graph || !Array.isArray(node?.outputs)) return;
  node.outputs.forEach((output, index) => {
    for (const linkId of output?.links || []) {
      const link = graph.links?.[linkId];
      if (link) link.origin_slot = index;
    }
  });
}

function removeReferencePackStaleOutputs(node) {
  if (!Array.isArray(node?.outputs)) return false;
  const removeSlots = [];
  for (let index = 0; index < node.outputs.length; index++) {
    const output = node.outputs[index];
    if (output?.name === "summary" || output?.type === "STRING") {
      removeSlots.push(index);
    }
  }
  for (const slot of removeSlots.reverse()) {
    removeOutputAt(node, slot);
  }
  if (removeSlots.length > 0) {
    updateOutputLinkOrigins(node);
    node._widgetSlotsDirty = true;
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
  }
  return false;
}

function rebuildReferencePackInputs(node, desiredNames) {
  const desired = new Set(desiredNames);
  const desiredOrder = new Map(desiredNames.map((name, index) => [name, index]));
  const seen = new Set();
  const removeSlots = [];

  for (let index = 0; index < (node.inputs || []).length; index++) {
    const input = node.inputs[index];
    if (!isReferencePackManagedInputName(input?.name)) continue;
    if (!desired.has(input.name) || seen.has(input.name)) {
      removeSlots.push(index);
    } else {
      seen.add(input.name);
    }
  }

  for (const slot of removeSlots.reverse()) {
    removeInputAt(node, slot);
  }

  let changed = removeSlots.length > 0;
  for (const name of desiredNames) {
    if (!findInputByName(node, name)) {
      if (typeof node.addInput === "function") {
        node.addInput(name, "IMAGE");
      } else {
        node.inputs ||= [];
        node.inputs.push(makeReferencePackInput(name));
      }
      changed = true;
    }
  }

  const managed = [];
  const unmanaged = [];
  for (const input of node.inputs || []) {
    if (!isReferencePackManagedInputName(input?.name)) {
      unmanaged.push(input);
      continue;
    }
    if (!desired.has(input.name)) continue;
    applyLabel(input, translationForInput("SCAIL2ReferencePack", input.name));
    input.type ||= "IMAGE";
    managed.push(input);
  }
  managed.sort((a, b) => desiredOrder.get(a.name) - desiredOrder.get(b.name));

  const sorted = [...managed, ...unmanaged];
  node.inputs ||= [];
  const orderChanged = sorted.length !== node.inputs.length || sorted.some((input, index) => input !== node.inputs[index]);
  if (changed || orderChanged) {
    node.inputs.splice(0, node.inputs.length, ...sorted);
    updateInputLinkTargets(node);
    node._widgetSlotsDirty = true;
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
  }
  return false;
}

function ensureOptions(widget) {
  widget.options ||= {};
  if (widget._state) {
    widget._state.options ||= {};
  }
}

function setOption(widget, key, value) {
  ensureOptions(widget);
  widget.options[key] = value;
  if (widget._state) {
    widget._state.options[key] = value;
  }
}

function deleteOption(widget, key) {
  widget.options ||= {};
  delete widget.options[key];
  if (widget._state?.options) {
    delete widget._state.options[key];
  }
}

function storeWidgetDefaults(widget) {
  if (!widget || WIDGET_DEFAULTS.has(widget)) return;
  WIDGET_DEFAULTS.set(widget, {
    type: widget.type,
    computeSize: widget.computeSize,
    computeSizeDescriptor: Object.getOwnPropertyDescriptor(widget, "computeSize"),
    hidden: widget.hidden,
    optionsHidden: widget.options?.hidden,
    optionsCanvasOnly: widget.options?.canvasOnly,
  });
}

function restoreComputeSize(widget, defaults) {
  if (defaults.computeSizeDescriptor) {
    Object.defineProperty(widget, "computeSize", defaults.computeSizeDescriptor);
  } else {
    delete widget.computeSize;
  }
}

function hideComputeSize(widget) {
  Object.defineProperty(widget, "computeSize", {
    value: hiddenWidgetComputeSize,
    configurable: true,
    writable: true,
    enumerable: false,
  });
}

function setWidgetVisible(widget, visible) {
  if (!widget) return false;
  storeWidgetDefaults(widget);
  const defaults = WIDGET_DEFAULTS.get(widget);
  const wasHidden = widget.type === "hidden" || widget.hidden === true || widget.options?.hidden === true;
  if (visible) {
    widget.hidden = defaults.hidden ?? false;
    widget.type = defaults.type;
    restoreComputeSize(widget, defaults);
    if (widget.inputEl) widget.inputEl.style.display = "";
    if (widget.element) widget.element.style.display = "";
    if (defaults.optionsHidden === undefined) {
      deleteOption(widget, "hidden");
    } else {
      setOption(widget, "hidden", defaults.optionsHidden);
    }
    if (defaults.optionsCanvasOnly === undefined) {
      deleteOption(widget, "canvasOnly");
    } else {
      setOption(widget, "canvasOnly", defaults.optionsCanvasOnly);
    }
  } else {
    widget.hidden = true;
    widget.type = "hidden";
    hideComputeSize(widget);
    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
    setOption(widget, "hidden", true);
    setOption(widget, "canvasOnly", true);
  }
  if (widget._state) {
    widget._state.hidden = widget.hidden;
    widget._state.type = widget.type;
  }
  widget.triggerDraw?.();
  const isHidden = widget.type === "hidden" || widget.hidden === true || widget.options?.hidden === true;
  return wasHidden !== isHidden;
}

function refreshWidgetSnapshot(node) {
  if (!Array.isArray(node?.widgets)) return;
  try {
    node.widgets = [...node.widgets];
  } catch {
    // Old ComfyUI builds expose widgets as a plain mutable field.
  }
}

function isWidgetVisible(widget) {
  return widget && widget.type !== "hidden" && widget.hidden !== true && widget.options?.hidden !== true;
}

function collectReferencePackWidgets(node) {
  if (!node._scail2ReferencePackWidgets) {
    node._scail2ReferencePackWidgets = new Map();
  }
  for (const widget of node.widgets || []) {
    if (isReferencePackCountWidgetName(widget?.name) || isLegacyReferencePackRefCountWidgetName(widget?.name)) {
      node._scail2ReferencePackWidgets.set(widget.name, widget);
    }
  }
  return node._scail2ReferencePackWidgets;
}

function syncReferencePackWidgetList(node) {
  if (!Array.isArray(node?.widgets)) return false;
  const widgetMap = collectReferencePackWidgets(node);
  const managedNames = ["subject_count", "reference_count"];
  for (const name of managedNames) {
    const widget = widgetMap.get(name) || getWidget(node, name);
    if (!widget) continue;
    clampReferencePackWidgetValue(widget, name, widget.value);
    applyLabel(widget, translationForInput("SCAIL2ReferencePack", name));
    if (!isWidgetVisible(widget)) setWidgetVisible(widget, true);
  }
  for (const [name, widget] of widgetMap.entries()) {
    if (isLegacyReferencePackRefCountWidgetName(name) && isWidgetVisible(widget)) {
      setWidgetVisible(widget, false);
    }
  }

  const managedWidgets = managedNames.map((name) => widgetMap.get(name) || getWidget(node, name)).filter(Boolean);
  const otherWidgets = (node.widgets || []).filter((widget) => (
    !isReferencePackCountWidgetName(widget?.name) && !isLegacyReferencePackRefCountWidgetName(widget?.name)
  ));
  const nextWidgets = [...managedWidgets, ...otherWidgets];
  const changed = nextWidgets.length !== node.widgets.length || nextWidgets.some((widget, index) => widget !== node.widgets[index]);
  if (changed) {
    node.widgets = nextWidgets;
    node._widgets = nextWidgets;
    node._widgetSlotsDirty = true;
  }
  return changed;
}

function fitNode(node, options = {}) {
  if (!node || node.flags?.collapsed) return;
  const width = Math.max(node.size?.[0] || 300, options.minWidth || 300);
  const size = node.computeSize?.([width, node.size?.[1] || 0]) || [width, 120];
  const height = options.height ?? size[1];
  node.setSize([Math.max(width, size[0]), height]);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function updateFitVideoWidgets(node) {
  const resolution = String(getWidget(node, "resolution")?.value || "512p");
  const custom = resolution === "custom";
  const changedWidth = setWidgetVisible(getWidget(node, "custom_width"), custom);
  const changedHeight = setWidgetVisible(getWidget(node, "custom_height"), custom);
  if (changedWidth || changedHeight) {
    refreshWidgetSnapshot(node);
  }
  fitNode(node);
}

function setupFitVideoNode(node) {
  if (FIT_VIDEO_SETUP.has(node)) return;
  FIT_VIDEO_SETUP.add(node);

  const resolutionWidget = getWidget(node, "resolution");
  if (resolutionWidget) {
    const original = resolutionWidget.callback;
    resolutionWidget.callback = function () {
      original?.apply(this, arguments);
      updateFitVideoWidgets(node);
    };
  }

  updateFitVideoWidgets(node);
}

function migrateLegacyReferencePackInputs(node) {
  if (!Array.isArray(node?.inputs)) return false;

  const inputs = [...node.inputs];
  const legacy = inputs
    .map((input, index) => {
      const match = /^subject_(\d+)_image_(\d+)$/.exec(String(input?.name || ""));
      return match ? { input, index, subject: Number(match[1]), image: Number(match[2]) } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.index - b.index);
  if (legacy.length === 0) return false;

  const occupied = new Set(inputs.map((input) => String(input?.name || "")));
  const extras = [];
  let maxSubject = 1;
  let changed = false;

  for (const item of legacy) {
    maxSubject = Math.max(maxSubject, item.subject);
    if (item.image === 1) {
      const targetName = `subject_${item.subject}_image`;
      const target = findInputByName(node, targetName);
      if (!target) {
        occupied.delete(item.input.name);
        item.input.name = targetName;
        occupied.add(targetName);
        applyLabel(item.input, translationForInput("SCAIL2ReferencePack", targetName));
        changed = true;
      } else if (target !== item.input && target.link == null && item.input.link != null) {
        target.link = item.input.link;
      }
    } else {
      extras.push(item.input);
    }
  }

  let nextReference = 1;
  for (const input of extras) {
    while (nextReference <= MAX_MIXED_REFERENCE_IMAGES && occupied.has(`reference_${nextReference}`)) {
      nextReference += 1;
    }
    if (nextReference > MAX_MIXED_REFERENCE_IMAGES) break;
    const targetName = `reference_${nextReference}`;
    occupied.delete(input.name);
    input.name = targetName;
    occupied.add(targetName);
    applyLabel(input, translationForInput("SCAIL2ReferencePack", targetName));
    nextReference += 1;
    changed = true;
  }

  const inferredReferences = Math.min(
    MAX_MIXED_REFERENCE_IMAGES,
    (node.inputs || []).filter((input) => /^reference_\d+$/.test(String(input?.name || ""))).length
  );
  const subjectWidget = getWidget(node, "subject_count");
  const referenceWidget = getWidget(node, "reference_count");
  if (subjectWidget) subjectWidget.value = clampInt(Math.max(Number(subjectWidget.value) || 1, maxSubject), 1, MAX_REFERENCE_SUBJECTS);
  if (referenceWidget) referenceWidget.value = inferredReferences;

  if (changed) {
    updateInputLinkTargets(node);
    node._widgetSlotsDirty = true;
  }
  return changed;
}

function desiredReferencePackInputNames(node) {
  const subjectCount = clampInt(getWidget(node, "subject_count")?.value ?? 1, 1, MAX_REFERENCE_SUBJECTS);
  const referenceCount = clampInt(getWidget(node, "reference_count")?.value ?? 0, 0, MAX_MIXED_REFERENCE_IMAGES);
  const names = [];
  for (let subject = 1; subject <= subjectCount; subject++) {
    names.push(`subject_${subject}_image`);
  }
  for (let reference = 1; reference <= referenceCount; reference++) {
    names.push(`reference_${reference}`);
  }
  names.push("scene_image");
  return names;
}

function referencePackSignature(node, desiredNames) {
  const subjectCount = clampInt(getWidget(node, "subject_count")?.value ?? 1, 1, MAX_REFERENCE_SUBJECTS);
  const referenceCount = clampInt(getWidget(node, "reference_count")?.value ?? 0, 0, MAX_MIXED_REFERENCE_IMAGES);
  return `${subjectCount}|${referenceCount}|${desiredNames.join(",")}`;
}

function updateReferencePackWidgets(node, options = {}) {
  if (node?._scail2ReferencePackUpdating) return false;
  node._scail2ReferencePackUpdating = true;
  try {
    const migrated = migrateLegacyReferencePackInputs(node);
    const desiredNames = desiredReferencePackInputNames(node);
    const signature = referencePackSignature(node, desiredNames);
    if (!options.force && !migrated && node._scail2ReferencePackSignature === signature) return false;
    node._scail2ReferencePackSignature = signature;

    const outputsChanged = removeReferencePackStaleOutputs(node);
    const changed = syncReferencePackWidgetList(node);
    const inputsChanged = rebuildReferencePackInputs(node, desiredNames);
    applyEasyNodeInstanceTranslation(node);
    if (outputsChanged || changed || inputsChanged) {
      refreshWidgetSnapshot(node);
    }
    fitNode(node, { minWidth: 300 });
    return migrated || outputsChanged || changed || inputsChanged;
  } finally {
    node._scail2ReferencePackUpdating = false;
  }
}

function setupReferencePackNode(node) {
  if (REFERENCE_PACK_SETUP.has(node)) return;
  REFERENCE_PACK_SETUP.add(node);
  applyEasyNodeInstanceTranslation(node);

  const widgetNames = ["subject_count", "reference_count"];
  for (const name of widgetNames) {
    const widget = getWidget(node, name);
    if (!widget) continue;
    const original = widget.callback;
    widget.callback = function (value) {
      original?.apply(this, arguments);
      clampReferencePackWidgetValue(widget, name, value);
      updateReferencePackWidgets(node);
    };
  }

  updateReferencePackWidgets(node, { force: true });
}

function updateSimpleVideoWidgets(node) {
  const advanced = Boolean(getWidget(node, "advanced")?.value);
  const longVideoMode = String(getWidget(node, "long_video_mode")?.value || "chunk");
  const visibleAdvanced = longVideoMode === "context_sampling" ? SIMPLE_CONTEXT_ADVANCED_WIDGETS : SIMPLE_CHUNK_ADVANCED_WIDGETS;
  const visibleSet = new Set(advanced ? [SIMPLE_ADVANCED_MODE_WIDGET, ...visibleAdvanced] : []);
  let changed = false;
  for (const name of SIMPLE_ALL_ADVANCED_WIDGETS) {
    changed = setWidgetVisible(getWidget(node, name), visibleSet.has(name)) || changed;
  }
  applyEasyNodeInstanceTranslation(node);
  if (changed) {
    refreshWidgetSnapshot(node);
  }
  reorderSimpleVideoWidgets(node);
  fitNode(node);
}

function reorderSimpleVideoWidgets(node) {
  if (!Array.isArray(node?.widgets)) return false;
  const order = new Map(SIMPLE_WIDGET_ORDER.map((name, index) => [name, index]));
  const managed = [];
  const other = [];
  for (const widget of node.widgets) {
    if (order.has(widget?.name)) {
      managed.push(widget);
    } else {
      other.push(widget);
    }
  }
  managed.sort((a, b) => order.get(a.name) - order.get(b.name));
  const nextWidgets = [...other, ...managed];
  const changed = nextWidgets.length !== node.widgets.length || nextWidgets.some((widget, index) => widget !== node.widgets[index]);
  if (changed) {
    node.widgets = nextWidgets;
    node._widgets = nextWidgets;
    node._widgetSlotsDirty = true;
    refreshWidgetSnapshot(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }
  return changed;
}

function setupSimpleVideoNode(node) {
  if (SIMPLE_VIDEO_SETUP.has(node)) return;
  SIMPLE_VIDEO_SETUP.add(node);
  applyEasyNodeInstanceTranslation(node);
  localizeComboWidget(getWidget(node, "mode"), "mode");
  localizeComboWidget(getWidget(node, "long_video_mode"), "long_video_mode");

  const advancedWidget = getWidget(node, "advanced");
  if (advancedWidget) {
    const original = advancedWidget.callback;
    advancedWidget.callback = function () {
      original?.apply(this, arguments);
      updateSimpleVideoWidgets(node);
    };
  }

  const modeWidget = getWidget(node, "long_video_mode");
  if (modeWidget) {
    const original = modeWidget.callback;
    modeWidget.callback = function () {
      original?.apply(this, arguments);
      updateSimpleVideoWidgets(node);
    };
  }

  updateSimpleVideoWidgets(node);
}

function repairSimpleVideoWidgetOrder(node, config) {
  const values = config?.widgets_values;
  if (!Array.isArray(values)) return;
  const modeIndex = values.findIndex((value) => value === "replacement" || value === "animation");
  if (modeIndex < 0) return;
  const widgetA = values[modeIndex + 1];
  const widgetB = values[modeIndex + 2];
  const advancedWidget = getWidget(node, "advanced");
  const longVideoModeWidget = getWidget(node, "long_video_mode");
  if (typeof widgetA === "boolean" && (widgetB === "chunk" || widgetB === "context_sampling")) {
    if (advancedWidget) advancedWidget.value = widgetA;
    if (longVideoModeWidget) longVideoModeWidget.value = widgetB;
  } else if ((widgetA === "chunk" || widgetA === "context_sampling") && typeof widgetB === "boolean") {
    if (advancedWidget) advancedWidget.value = widgetB;
    if (longVideoModeWidget) longVideoModeWidget.value = widgetA;
  } else if (typeof widgetA === "boolean" && typeof widgetB === "number") {
    if (advancedWidget) advancedWidget.value = widgetA;
    if (longVideoModeWidget) longVideoModeWidget.value = "chunk";
    const maxFramesWidget = getWidget(node, "max_frames");
    const chunkFramesWidget = getWidget(node, "chunk_frames");
    const overlapFramesWidget = getWidget(node, "overlap_frames");
    if (maxFramesWidget) maxFramesWidget.value = widgetB;
    if (chunkFramesWidget && typeof values[modeIndex + 3] === "number") chunkFramesWidget.value = values[modeIndex + 3];
    if (overlapFramesWidget && typeof values[modeIndex + 4] === "number") overlapFramesWidget.value = values[modeIndex + 4];
  }
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    applyEasyNodeDataTranslation(nodeData);
    if (
      nodeData.name !== "SCAIL2FitVideo"
      && nodeData.name !== "SCAIL2ReferencePack"
      && nodeData.name !== "SCAIL2ReferenceSAMBuilder"
      && nodeData.name !== "SCAIL2SimpleVideo"
    ) return;

    if (nodeData.name === "SCAIL2ReferencePack") {
      trimReferencePackNodeDataInputs(nodeData);
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      if (nodeData.name === "SCAIL2FitVideo") setupFitVideoNode(this);
      if (nodeData.name === "SCAIL2ReferencePack") setupReferencePackNode(this);
      if (nodeData.name === "SCAIL2ReferenceSAMBuilder") applyEasyNodeInstanceTranslation(this);
      if (nodeData.name === "SCAIL2SimpleVideo") setupSimpleVideoNode(this);
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      const result = onAdded?.apply(this, arguments);
      if (nodeData.name === "SCAIL2ReferencePack") {
        setupReferencePackNode(this);
        updateReferencePackWidgets(this);
      }
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      if (nodeData.name === "SCAIL2FitVideo") {
        setupFitVideoNode(this);
        updateFitVideoWidgets(this);
      }
      if (nodeData.name === "SCAIL2ReferencePack") {
        setupReferencePackNode(this);
        updateReferencePackWidgets(this);
      }
      if (nodeData.name === "SCAIL2ReferenceSAMBuilder") {
        applyEasyNodeInstanceTranslation(this);
      }
      if (nodeData.name === "SCAIL2SimpleVideo") {
        repairSimpleVideoWidgetOrder(this, arguments[0]);
        setupSimpleVideoNode(this);
        updateSimpleVideoWidgets(this);
      }
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (name) {
      const result = onWidgetChanged?.apply(this, arguments);
      const widgetName = typeof name === "string" ? name : name?.name;
      if (nodeData.name === "SCAIL2FitVideo" && widgetName === "resolution") {
        updateFitVideoWidgets(this);
      }
      if (nodeData.name === "SCAIL2SimpleVideo" && (widgetName === "advanced" || widgetName === "long_video_mode")) {
        updateSimpleVideoWidgets(this);
      }
      return result;
    };
  },
});
