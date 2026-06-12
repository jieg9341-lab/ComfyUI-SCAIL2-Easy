import { app } from "../../scripts/app.js";

const EXT_NAME = "Comfy.SCAIL2.Easy";
const SIMPLE_ADVANCED_WIDGETS = ["max_frames", "chunk_frames", "overlap_frames"];
const WIDGET_DEFAULTS = new WeakMap();
const FIT_VIDEO_SETUP = new WeakSet();
const SIMPLE_VIDEO_SETUP = new WeakSet();
const hiddenWidgetComputeSize = () => [0, -4];
const EASY_TRANSLATIONS = {
  SCAIL2SimpleVideo: {
    inputs: {
      advanced: "\u9ad8\u7ea7\u53c2\u6570",
      max_frames: "\u6700\u5927\u5e27\u6570",
      chunk_frames: "\u6bcf\u6bb5\u5e27\u6570",
      overlap_frames: "\u91cd\u53e0\u5e27\u6570",
    },
  },
};

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
  return EASY_TRANSLATIONS[className]?.inputs?.[name];
}

function applyEasyNodeDataTranslation(nodeData) {
  if (!isChineseLocale()) return;
  const translation = EASY_TRANSLATIONS[nodeData?.name];
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

function applyEasyNodeInstanceTranslation(node) {
  if (!isChineseLocale()) return;
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

function fitNode(node) {
  if (!node?.graph || node.flags?.collapsed) return;
  const size = node.computeSize();
  node.setSize([Math.max(node.size?.[0] || 300, size[0]), size[1]]);
  node.setDirtyCanvas?.(true, true);
  node.graph.setDirtyCanvas(true, true);
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

function updateSimpleVideoWidgets(node) {
  const advanced = Boolean(getWidget(node, "advanced")?.value);
  let changed = false;
  for (const name of SIMPLE_ADVANCED_WIDGETS) {
    changed = setWidgetVisible(getWidget(node, name), advanced) || changed;
  }
  applyEasyNodeInstanceTranslation(node);
  if (changed) {
    refreshWidgetSnapshot(node);
  }
  fitNode(node);
}

function setupSimpleVideoNode(node) {
  if (SIMPLE_VIDEO_SETUP.has(node)) return;
  SIMPLE_VIDEO_SETUP.add(node);
  applyEasyNodeInstanceTranslation(node);

  const advancedWidget = getWidget(node, "advanced");
  if (advancedWidget) {
    const original = advancedWidget.callback;
    advancedWidget.callback = function () {
      original?.apply(this, arguments);
      updateSimpleVideoWidgets(node);
    };
  }

  updateSimpleVideoWidgets(node);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    applyEasyNodeDataTranslation(nodeData);
    if (nodeData.name !== "SCAIL2FitVideo" && nodeData.name !== "SCAIL2SimpleVideo") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      if (nodeData.name === "SCAIL2FitVideo") setupFitVideoNode(this);
      if (nodeData.name === "SCAIL2SimpleVideo") setupSimpleVideoNode(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      if (nodeData.name === "SCAIL2FitVideo") {
        setupFitVideoNode(this);
        updateFitVideoWidgets(this);
      }
      if (nodeData.name === "SCAIL2SimpleVideo") {
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
      if (nodeData.name === "SCAIL2SimpleVideo" && widgetName === "advanced") {
        updateSimpleVideoWidgets(this);
      }
      return result;
    };
  },
});
