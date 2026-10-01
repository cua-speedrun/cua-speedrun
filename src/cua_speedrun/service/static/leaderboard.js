/* Interactive exploration of pre-rendered measurements. Public exports select
   a dataset; the evaluator dashboard retains its own comparison boundaries. */

(function () {
  const root = document.querySelector("[data-results-dashboard]");
  if (!root) return;
  const staticMode = root.dataset.static === "true";

  const dataNode = root.querySelector("[data-results-data]");
  // Public variants get independent filters, curves, colors and tooltips.
  // The embedded source records retain the original model name separately.
  const entries = JSON.parse(dataNode?.textContent || "[]").map((entry) => ({
    ...entry, model: entry.series || entry.model,
  }));
  const rows = new Map(
    Array.from(root.querySelectorAll("[data-entry-id]")).map((row) => [
      Number(row.dataset.entryId),
      row,
    ])
  );
  const body = root.querySelector("[data-results-body]");
  const visibleCount = root.querySelector("[data-visible-count]");
  const noResults = root.querySelector("[data-no-results]");
  const familyFilters = root.querySelector("[data-family-filters]");
  const effortFilters = root.querySelector("[data-effort-filters]");
  const showAllEfforts = root.querySelector("[data-show-all-efforts]");
  const chart = root.querySelector("[data-results-chart]");
  const chartTooltip = root.querySelector("[data-chart-tooltip]");
  const modelLegend = root.querySelector("[data-model-legend]");
  const rankingDescription = root.querySelector("[data-ranking-description]");
  const curveNote = root.querySelector("[data-curve-note]");
  const frontierSummary = root.querySelector("[data-frontier-summary]");
  const baselineSummary = root.querySelector("[data-baseline-summary]");
  const baselineLabel = root.querySelector("[data-baseline-label]");
  const volumeChart = root.querySelector("[data-results-volume]");
  const volumeFrame = root.querySelector("[data-volume-frame]");
  const volumeTooltip = root.querySelector("[data-volume-tooltip]");
  const volumeLegend = root.querySelector("[data-volume-legend]");
  const volumeFrontierCount = root.querySelector("[data-frontier3d-count]");
  const volumeCostCount = root.querySelector("[data-cost3d-count]");
  const volumeNote = root.querySelector("[data-volume-note]");
  const surfaceLabel = root.querySelector("[data-surface-label]");
  const surfaceCache = new Map();
  const initialParams = new URLSearchParams(window.location.search);

  const palette = [
    "#72DCC8", "#9B82F3", "#83B9F4", "#527BD4", "#C7D9F7",
    "#FF986F", "#ED91B1", "#A4D968", "#F2C14E", "#DA8AC5",
    "#F18462", "#6DD6A8", "#C99CF4", "#F0A6CA",
  ];
  const familyNames = Array.from(new Set(entries.map((entry) => entry.model)));
  const effortNames = Array.from(new Set(entries.map((entry) => entry.effort))).sort(
    (left, right) => {
      const a = entries.find((entry) => entry.effort === left)?.effort_rank ?? -1;
      const b = entries.find((entry) => entry.effort === right)?.effort_rank ?? -1;
      return a - b || left.localeCompare(right);
    }
  );
  const colors = new Map(familyNames.map((name, index) => [name, palette[index % palette.length]]));
  const state = {
    view: "leaderboard",
    ranking: "performance",
    curve: "average_time",
    access: "all",
    families: new Set(familyNames),
    efforts: new Set(effortNames),
    showAllEfforts: true,
    volumeYaw: -0.72,
    volumePitch: 0.55,
    surfaceMode: ["dominance", "smooth"].includes(initialParams.get("surface"))
      ? initialParams.get("surface") : "smooth",
  };

  function countFor(key, value) {
    return entries.filter((entry) => entry[key] === value).length;
  }

  function filterCheck(kind, value, label, count, color) {
    const wrapper = document.createElement("label");
    wrapper.className = "filter-check";
    if (color) wrapper.style.setProperty("--model-color", color);
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.dataset.filterKind = kind;
    input.value = value;
    const swatch = color ? document.createElement("i") : null;
    const text = document.createElement("span");
    text.textContent = label;
    const total = document.createElement("small");
    total.textContent = String(count);
    wrapper.append(input);
    if (swatch) wrapper.append(swatch);
    wrapper.append(text, total);
    return wrapper;
  }

  familyNames.forEach((family) => {
    familyFilters.append(filterCheck(
      "family", family, family, countFor("model", family), colors.get(family)
    ));
  });
  effortNames.forEach((effort) => {
    effortFilters.append(filterCheck(
      "effort", effort, effort, countFor("effort", effort)
    ));
  });
  entries.forEach((entry) => {
    rows.get(entry.entry_id)?.style.setProperty("--model-color", colors.get(entry.model));
  });

  function baseFilteredEntries() {
    let filtered = entries.filter((entry) =>
      (state.access === "all" || entry.model_type === state.access) &&
      state.families.has(entry.model) &&
      state.efforts.has(entry.effort)
    );
    if (state.showAllEfforts) return filtered;

    const highest = new Map();
    filtered.forEach((entry) => {
      const previous = highest.get(entry.model);
      if (!previous ||
          entry.effort_rank > previous.effort_rank ||
          (entry.effort_rank === previous.effort_rank && entry.performance > previous.performance) ||
          (entry.effort_rank === previous.effort_rank && entry.performance === previous.performance &&
           entry.time_per_task_sec < previous.time_per_task_sec)) {
        highest.set(entry.model, entry);
      }
    });
    return Array.from(highest.values());
  }

  function rankedEntries() {
    let filtered = baseFilteredEntries();
    if (!staticMode && state.ranking !== "performance") {
      filtered = filtered.filter((entry) => entry.performance > 0.8);
    }
    if (!staticMode && state.ranking === "cost") {
      filtered = filtered.filter((entry) => Number.isFinite(entry.cost_usd));
    }
    const sorters = {
      performance: (a, b) => b.performance - a.performance || a.time_per_task_sec - b.time_per_task_sec,
      time: (a, b) => a.time_per_task_sec - b.time_per_task_sec || b.performance - a.performance,
      cost: (a, b) => (a.cost_usd ?? Infinity) - (b.cost_usd ?? Infinity) || b.performance - a.performance || a.time_per_task_sec - b.time_per_task_sec,
    };
    return filtered.sort(sorters[state.ranking]);
  }

  function renderTable() {
    const ranked = rankedEntries();
    const visibleIds = new Set(ranked.map((entry) => entry.entry_id));
    ranked.forEach((entry, index) => {
      const row = rows.get(entry.entry_id);
      if (!row) return;
      row.hidden = false;
      row.querySelector("[data-rank]").textContent = String(index + 1).padStart(2, "0");
      body.append(row);
    });
    entries.forEach((entry) => {
      if (visibleIds.has(entry.entry_id)) return;
      const row = rows.get(entry.entry_id);
      if (row) {
        row.hidden = true;
        body.append(row);
      }
    });
    noResults.hidden = ranked.length !== 0;
    const descriptions = staticMode ? {
      performance: "All included results, ordered by average performance.",
      time: "All included results, ordered by average time per task.",
      cost: "Ordered by cost per task; unknown costs remain at the bottom.",
    } : {
      performance: "Every published result, ordered by average performance.",
      time: "Results above 80% performance, ordered by time per task.",
      cost: "Results above 80% performance with reported cost, ordered by cost per task.",
    };
    rankingDescription.textContent = descriptions[state.ranking];
    if (state.view === "leaderboard") visibleCount.textContent = String(ranked.length);
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function svgElement(name, attributes = {}, text = "") {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text) element.textContent = text;
    return element;
  }

  function niceStep(rough) {
    if (!Number.isFinite(rough) || rough <= 0) return 1;
    const power = 10 ** Math.floor(Math.log10(rough));
    const value = rough / power;
    const nice = value <= 1 ? 1 : value <= 2 ? 2 : value <= 5 ? 5 : 10;
    return nice * power;
  }

  const baselineNode = root.querySelector("[data-results-baseline]");
  const referenceBaseline = baselineNode ? JSON.parse(baselineNode.textContent) : null;
  if (referenceBaseline) {
    baselineSummary.hidden = false;
    baselineLabel.textContent = referenceBaseline.label;
  }

  const DAY_MS = 86_400_000;
  entries.forEach((entry) => {
    const timestamp = Date.parse(`${entry.release_date || ""}T00:00:00Z`);
    entry.release_date_value = Number.isFinite(timestamp) ? timestamp / DAY_MS : null;
  });

  function formatReleaseDate(value, compact = false) {
    const date = new Date(value * DAY_MS);
    return new Intl.DateTimeFormat("en-US", {
      month: compact ? "short" : "short",
      day: compact ? undefined : "numeric",
      year: "numeric",
      timeZone: "UTC",
    }).format(date);
  }

  const curveMetrics = {
    average_time: {
      field: "average_time_per_task_sec",
      axisLabel: "Average time per task (seconds)",
      noun: "average time",
      lowerBetter: true,
      formatTick: (value) => `${Math.round(value).toLocaleString()}s`,
      formatValue: (value) => `${value.toFixed(1)}s average / task`,
    },
    median_time: {
      field: "median_time_per_task_sec",
      axisLabel: "Median time per task (seconds)",
      noun: "median time",
      lowerBetter: true,
      formatTick: (value) => `${Math.round(value).toLocaleString()}s`,
      formatValue: (value) => `${value.toFixed(1)}s median / task`,
    },
    cost: {
      field: "cost_usd",
      axisLabel: "Cost per task (USD)",
      noun: "cost",
      lowerBetter: true,
      formatTick: (value) => `$${value < 10 ? value.toFixed(2) : value.toFixed(0)}`,
      formatValue: (value) => `$${value.toFixed(4)} / task`,
    },
    steps: {
      field: "steps_per_task",
      axisLabel: "Average steps per task (action batches)",
      noun: "step count",
      lowerBetter: true,
      formatTick: (value) => value.toFixed(value < 10 ? 1 : 0),
      formatValue: (value) => `${value.toFixed(2)} steps / task · one action batch per step`,
    },
    tool_calls: {
      field: "tool_calls_per_task",
      axisLabel: "Average tool calls per task (individual environment actions)",
      noun: "tool-call count",
      description: "Each action within a batch counts separately, including waits.",
      lowerBetter: true,
      formatTick: (value) => value.toFixed(value < 10 ? 1 : 0),
      formatValue: (value) => `${value.toFixed(2)} tool calls / task · individual environment actions`,
    },
    turns: {
      field: "turns_per_task",
      axisLabel: "Average model turns per task",
      noun: "turn count",
      lowerBetter: true,
      formatTick: (value) => value.toFixed(value < 10 ? 1 : 0),
      formatValue: (value) => `${value.toFixed(2)} turns / task`,
    },
    release_date: {
      field: "release_date_value",
      axisLabel: "Model release date",
      noun: "release date",
      lowerBetter: true,
      date: true,
      formatTick: (value) => formatReleaseDate(value, true),
      formatValue: (value) => `Released ${formatReleaseDate(value)}`,
    },
    output_tokens: {
      field: "average_output_tokens",
      axisLabel: "Average output length (tokens per model response)",
      noun: "output length",
      lowerBetter: true,
      formatTick: (value) => Math.round(value).toLocaleString(),
      formatValue: (value) => `${value.toFixed(1)} output tokens / model response`,
    },
  };

  function paretoPoints(points, metric) {
    const sorted = [...points].sort((a, b) =>
      a[metric] - b[metric] || b.performance - a.performance
    );
    const frontier = [];
    let best = -Infinity;
    sorted.forEach((point) => {
      if (point.performance > best) {
        frontier.push(point);
        best = point.performance;
      }
    });
    return frontier;
  }

  function showTooltip(event, entry) {
    const frame = chart.closest(".chart-frame");
    const frameBox = frame.getBoundingClientRect();
    const targetBox = event.target.getBoundingClientRect();
    const clientX = Number.isFinite(event.clientX) ? event.clientX : targetBox.right;
    const clientY = Number.isFinite(event.clientY) ? event.clientY : targetBox.bottom;
    chartTooltip.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = entry.model;
    const details = document.createElement("span");
    const metric = curveMetrics[state.curve];
    const xText = metric.formatValue(entry[metric.field]);
    const effort = document.createElement("b");
    effort.textContent = entry.effort;
    details.append(
      effort,
      document.createElement("br"),
      document.createTextNode(`${(entry.performance * 100).toFixed(2)}% performance`),
      document.createElement("br"),
      document.createTextNode(xText),
    );
    if (metric.field === "cost_usd") appendCostDetails(details, entry);
    chartTooltip.append(title, details);
    chartTooltip.hidden = false;
    chartTooltip.style.left = `${clientX - frameBox.left + frame.scrollLeft + 12}px`;
    chartTooltip.style.top = `${clientY - frameBox.top + 12}px`;
  }

  function renderChart() {
    const metricDefinition = curveMetrics[state.curve];
    const metric = metricDefinition.field;
    const filtered = baseFilteredEntries();
    const points = filtered.filter((entry) => Number.isFinite(entry[metric]));
    const metricReference = referenceBaseline?.metric_references?.[state.curve] || null;
    const referencePerformance = Number.isFinite(referenceBaseline?.performance)
      ? referenceBaseline.performance
      : null;
    const width = 1080;
    const height = 620;
    const margin = { left: 76, right: 34, top: 34, bottom: 72 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    chart.replaceChildren();

    if (!points.length) {
      chart.append(svgElement("text", {
        class: "empty-chart", x: width / 2, y: height / 2, "text-anchor": "middle",
      }, `No results report ${metricDefinition.noun} for the current filters.`));
      modelLegend.replaceChildren();
      chartTooltip.hidden = true;
      frontierSummary.hidden = true;
      curveNote.textContent = metricDefinition.description || `No reported ${metricDefinition.noun} for the current filters.`;
      if (state.view === "curves") visibleCount.textContent = "0";
      return;
    }

    const minX = Math.min(...points.map((point) => point[metric]));
    const measuredMaxX = Math.max(...points.map((point) => point[metric]));
    const maxX = Number.isFinite(metricReference?.value)
      ? Math.max(measuredMaxX, metricReference.value)
      : measuredMaxX;
    let xFloor = 0;
    let xCeiling;
    let xTicks;
    if (metricDefinition.date) {
      const span = Math.max(maxX - minX, 30);
      xFloor = minX - span * 0.06;
      xCeiling = maxX + span * 0.06;
      xTicks = Array.from({ length: 6 }, (_, index) =>
        xFloor + (xCeiling - xFloor) * index / 5
      );
    } else {
      const step = niceStep(Math.max(maxX, metric === "cost_usd" ? 0.1 : 1) * 1.08 / 5);
      xCeiling = Math.max(step, Math.ceil(Math.max(maxX, step) * 1.06 / step) * step);
      xTicks = Array.from(
        { length: Math.round(xCeiling / step) + 1 },
        (_, index) => index * step,
      );
    }
    const performanceValues = points.map((point) => point.performance);
    if (referencePerformance !== null) performanceValues.push(referencePerformance);
    const minimumPerformance = Math.min(...performanceValues);
    const maximumPerformance = Math.max(...performanceValues);
    let yFloor = Math.max(0, Math.floor((minimumPerformance * 100 - 5) / 10) * 10 / 100);
    let yCeiling = Math.min(1, Math.ceil((maximumPerformance * 100 + 4) / 5) * 5 / 100);
    if (yCeiling - yFloor < 0.2) yFloor = Math.max(0, yCeiling - 0.2);
    const x = (value) => margin.left + (value - xFloor) / (xCeiling - xFloor) * plotWidth;
    const y = (value) => margin.top + (yCeiling - value) / (yCeiling - yFloor) * plotHeight;

    const referencePerformanceY = referencePerformance === null
      ? null
      : y(referencePerformance);
    if (metricReference && referencePerformanceY !== null) {
      const referenceMetricX = x(metricReference.value);
      chart.append(svgElement("rect", {
        class: "reference-zone",
        x: margin.left,
        y: margin.top,
        width: Math.max(0, referenceMetricX - margin.left),
        height: Math.max(0, referencePerformanceY - margin.top),
      }));
      chart.append(svgElement("text", {
        class: "reference-zone-label",
        x: margin.left + 13,
        y: margin.top + 18,
      }, String(metricReference.region_label || "Reference frontier").toUpperCase()));
    }

    for (let index = 0; index <= 5; index += 1) {
      const value = yFloor + (yCeiling - yFloor) * index / 5;
      const py = y(value);
      chart.append(svgElement("line", {
        class: "chart-grid", x1: margin.left, x2: width - margin.right, y1: py, y2: py,
      }));
      chart.append(svgElement("text", {
        class: "chart-tick", x: margin.left - 13, y: py + 4, "text-anchor": "end",
      }, `${Math.round(value * 100)}%`));
    }
    xTicks.forEach((value) => {
      const px = x(value);
      chart.append(svgElement("line", {
        class: "chart-grid", x1: px, x2: px, y1: margin.top, y2: height - margin.bottom,
      }));
      chart.append(svgElement("text", {
        class: "chart-tick", x: px, y: height - margin.bottom + 24, "text-anchor": "middle",
      }, metricDefinition.formatTick(value)));
    });
    chart.append(svgElement("line", {
      class: "chart-axis", x1: margin.left, x2: width - margin.right,
      y1: height - margin.bottom, y2: height - margin.bottom,
    }));
    chart.append(svgElement("line", {
      class: "chart-axis", x1: margin.left, x2: margin.left,
      y1: margin.top, y2: height - margin.bottom,
    }));
    chart.append(svgElement("text", {
      class: "chart-label", x: margin.left + plotWidth / 2, y: height - 16, "text-anchor": "middle",
    }, metricDefinition.axisLabel));
    const yLabel = svgElement("text", {
      class: "chart-label", x: 18, y: margin.top + plotHeight / 2, "text-anchor": "middle",
      transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`,
    }, "Average performance");
    chart.append(yLabel);

    if (referencePerformanceY !== null) {
      chart.append(svgElement("line", {
        class: "reference-baseline reference-performance-baseline",
        x1: margin.left,
        x2: width - margin.right,
        y1: referencePerformanceY,
        y2: referencePerformanceY,
      }));
      chart.append(svgElement("text", {
        class: "reference-baseline-label",
        x: width - margin.right - 8,
        y: referencePerformanceY - 8,
        "text-anchor": "end",
      }, String(
        referenceBaseline.performance_label || `${referencePerformance * 100}% reference`,
      ).toUpperCase()));
    }
    if (metricReference) {
      const referenceMetricX = x(metricReference.value);
      chart.append(svgElement("line", {
        class: "reference-baseline reference-metric-baseline",
        x1: referenceMetricX,
        x2: referenceMetricX,
        y1: margin.top,
        y2: height - margin.bottom,
      }));
      chart.append(svgElement("text", {
        class: "reference-baseline-label",
        x: referenceMetricX + 8,
        y: height - margin.bottom - 10,
      }, String(metricReference.label).toUpperCase()));
    }

    familyNames.forEach((family) => {
      const familyPoints = points
        .filter((entry) => entry.model === family)
        .sort((a, b) => a.effort_rank - b.effort_rank || a.performance - b.performance);
      if (familyPoints.length < 2) return;
      const path = familyPoints.map((entry, index) =>
        `${index ? "L" : "M"} ${x(entry[metric]).toFixed(1)} ${y(entry.performance).toFixed(1)}`
      ).join(" ");
      chart.append(svgElement("path", {
        class: "family-path", d: path, stroke: colors.get(family), "data-series": family,
      }));
    });

    const frontier = metricDefinition.lowerBetter ? paretoPoints(points, metric) : [];
    if (frontier.length > 1) {
      const path = frontier.map((entry, index) =>
        `${index ? "L" : "M"} ${x(entry[metric]).toFixed(1)} ${y(entry.performance).toFixed(1)}`
      ).join(" ");
      chart.append(svgElement("path", { class: "pareto-path", d: path }));
    }
    const frontierCoordinates = new Set(
      frontier.map((entry) => `${entry[metric]}:${entry.performance}`)
    );

    const pointLabelBounds = [];
    points.forEach((entry, index) => {
      const px = x(entry[metric]);
      const py = y(entry.performance);
      if (frontierCoordinates.has(`${entry[metric]}:${entry.performance}`)) {
        chart.append(svgElement("circle", {
          class: "pareto-ring", cx: px, cy: py, r: 10,
        }));
      }
      const point = svgElement("circle", {
        class: "result-point", cx: px, cy: py, r: 5.5, fill: colors.get(entry.model),
        tabindex: 0, role: "img", "aria-label": `${entry.model}, ${entry.effort}`,
      });
      point.style.animationDelay = `${Math.min(index * 18, 260)}ms`;
      point.addEventListener("pointerenter", (event) => showTooltip(event, entry));
      point.addEventListener("pointermove", (event) => showTooltip(event, entry));
      point.addEventListener("pointerleave", () => { chartTooltip.hidden = true; });
      point.addEventListener("focus", (event) => showTooltip(event, entry));
      point.addEventListener("blur", () => { chartTooltip.hidden = true; });
      if (!staticMode) {
        point.addEventListener("click", () => { window.location.href = `/entries/${entry.entry_id}`; });
      }
      chart.append(point);
      const label = svgElement("text", {
        class: "point-label", x: px + 9, y: py + (index % 2 ? 13 : -9),
        fill: colors.get(entry.model),
      }, entry.effort);
      chart.append(label);
      // Keep the usual placement unless two effort labels overlap.
      const defaultOffset = index % 2 ? 13 : -9;
      let labelY = py + defaultOffset;
      for (const offset of [defaultOffset, -defaultOffset, -25, 29, -41, 45]) {
        label.setAttribute("y", py + offset);
        const bounds = label.getBBox();
        if (bounds.y < margin.top || bounds.y + bounds.height > height - margin.bottom) continue;
        const overlaps = pointLabelBounds.some((other) =>
          bounds.x < other.x + other.width + 3 && bounds.x + bounds.width + 3 > other.x
          && bounds.y < other.y + other.height + 3 && bounds.y + bounds.height + 3 > other.y
        );
        if (!overlaps) {
          labelY = py + offset;
          break;
        }
      }
      label.setAttribute("y", labelY);
      pointLabelBounds.push(label.getBBox());
    });

    const visibleFamilies = familyNames.filter((family) =>
      points.some((entry) => entry.model === family)
    );
    modelLegend.replaceChildren(...visibleFamilies.map((family) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("i");
      swatch.className = "legend-swatch";
      swatch.style.setProperty("--model-color", colors.get(family));
      item.append(swatch, document.createTextNode(family));
      return item;
    }));
    const omitted = filtered.length - points.length;
    frontierSummary.hidden = !metricDefinition.lowerBetter;
    const omittedText = omitted
      ? ` ${omitted} ${omitted === 1 ? "result" : "results"} without reported `
        + `${metricDefinition.noun} ${omitted === 1 ? "is" : "are"} omitted.`
      : "";
    const referenceText = referenceBaseline
      ? ` ${referenceBaseline.label}: ${referenceBaseline.performance_label}`
        + `${metricReference ? ` and ${metricReference.label}` : ""} on the ${referenceBaseline.note}.`
      : "";
    const metricDescription = metricDefinition.description ? ` ${metricDefinition.description}` : "";
    curveNote.textContent = metricDefinition.date
      ? `The frontier traces models that set a new performance high when released.${referenceText}${omittedText}`
      : `Higher performance and lower ${metricDefinition.noun} are better.${metricDescription}${referenceText}${omittedText}`;
    if (state.view === "curves") visibleCount.textContent = String(points.length);
  }

  function paretoVolumePoints(points) {
    return points.filter((candidate) => !points.some((other) => {
      if (other === candidate) return false;
      const noWorse = other.cost_usd <= candidate.cost_usd &&
        other.time_per_task_sec <= candidate.time_per_task_sec &&
        other.performance >= candidate.performance;
      const strictlyBetter = other.cost_usd < candidate.cost_usd ||
        other.time_per_task_sec < candidate.time_per_task_sec ||
        other.performance > candidate.performance;
      return noWorse && strictlyBetter;
    }));
  }

  function convexHull2D(points) {
    const unique = Array.from(new Map(
      points.map((point) => [`${point.cost}:${point.time}`, point])
    ).values()).sort((left, right) => left.cost - right.cost || left.time - right.time);
    if (unique.length <= 2) return unique;
    const cross = (origin, left, right) =>
      (left.cost - origin.cost) * (right.time - origin.time) -
      (left.time - origin.time) * (right.cost - origin.cost);
    const half = (ordered) => {
      const result = [];
      ordered.forEach((point) => {
        while (result.length >= 2 &&
          cross(result[result.length - 2], result[result.length - 1], point) <= 0) {
          result.pop();
        }
        result.push(point);
      });
      return result;
    };
    const lower = half(unique);
    const upper = half([...unique].reverse());
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function interpolatedPerformance(cost, time, sources) {
    let weightedPerformance = 0;
    let totalWeight = 0;
    for (const source of sources) {
      const distanceSquared = (source.cost - cost) ** 2 + (source.time - time) ** 2;
      if (distanceSquared < 1e-10) return source.performance;
      const weight = 1 / distanceSquared;
      weightedPerformance += source.performance * weight;
      totalWeight += weight;
    }
    return weightedPerformance / totalWeight;
  }

  function paretoSurface(frontierPoints, resolution = 9) {
    const hull = convexHull2D(frontierPoints);
    if (hull.length < 3) return { hull, triangles: [] };
    const center = {
      cost: hull.reduce((sum, point) => sum + point.cost, 0) / hull.length,
      time: hull.reduce((sum, point) => sum + point.time, 0) / hull.length,
    };
    const sample = (origin, costEdge, timeEdge, costIndex, timeIndex) => {
      const costRatio = costIndex / resolution;
      const timeRatio = timeIndex / resolution;
      const cost = origin.cost + (costEdge.cost - origin.cost) * costRatio +
        (timeEdge.cost - origin.cost) * timeRatio;
      const time = origin.time + (costEdge.time - origin.time) * costRatio +
        (timeEdge.time - origin.time) * timeRatio;
      return {
        cost,
        time,
        performance: interpolatedPerformance(cost, time, frontierPoints),
      };
    };
    const triangles = [];
    hull.forEach((point, index) => {
      const next = hull[(index + 1) % hull.length];
      for (let costIndex = 0; costIndex < resolution; costIndex += 1) {
        for (let timeIndex = 0;
          timeIndex < resolution - costIndex;
          timeIndex += 1) {
          const origin = sample(center, point, next, costIndex, timeIndex);
          const alongCost = sample(center, point, next, costIndex + 1, timeIndex);
          const alongTime = sample(center, point, next, costIndex, timeIndex + 1);
          triangles.push([origin, alongCost, alongTime]);
          if (costIndex + timeIndex <= resolution - 2) {
            const diagonal = sample(
              center, point, next, costIndex + 1, timeIndex + 1
            );
            triangles.push([alongCost, diagonal, alongTime]);
          }
        }
      }
    });
    return { hull, triangles };
  }

  function surfaceColor(performance) {
    const mix = Math.max(0, Math.min(1, performance));
    const low = [91, 73, 162];
    const high = [114, 220, 200];
    return `rgb(${low.map((value, index) =>
      Math.round(value + (high[index] - value) * mix)
    ).join(", ")})`;
  }

  function appendCostDetails(details, entry) {
    if (!entry.cost_label) return;
    details.append(document.createElement("br"), document.createTextNode(entry.cost_label));
    if (Number.isFinite(entry.cost_range_min_usd)) {
      details.append(document.createElement("br"), document.createTextNode(
        `Cache range: $${entry.cost_range_min_usd.toFixed(4)}–$${entry.cost_range_max_usd.toFixed(4)}`
      ));
    }
  }

  function showVolumeTooltip(event, entry, frontier) {
    const frameBox = volumeFrame.getBoundingClientRect();
    const targetBox = event.target.getBoundingClientRect();
    const clientX = Number.isFinite(event.clientX) ? event.clientX : targetBox.right;
    const clientY = Number.isFinite(event.clientY) ? event.clientY : targetBox.bottom;
    volumeTooltip.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = entry.model;
    const details = document.createElement("span");
    const effort = document.createElement("b");
    effort.textContent = entry.effort;
    details.append(
      effort,
      document.createTextNode(frontier ? " · non-dominated" : " · dominated"),
      document.createElement("br"),
      document.createTextNode(`${(entry.performance * 100).toFixed(2)}% performance`),
      document.createElement("br"),
      document.createTextNode(`${entry.time_per_task_sec.toFixed(1)}s / task`),
      document.createElement("br"),
      document.createTextNode(`$${entry.cost_usd.toFixed(4)} / task`),
    );
    appendCostDetails(details, entry);
    volumeTooltip.append(title, details);
    volumeTooltip.hidden = false;
    volumeTooltip.style.left = `${clientX - frameBox.left + volumeFrame.scrollLeft + 13}px`;
    volumeTooltip.style.top = `${clientY - frameBox.top + 13}px`;
  }

  function renderVolume() {
    if (!volumeChart) return;
    const filtered = baseFilteredEntries();
    const points = filtered.filter((entry) => Number.isFinite(entry.cost_usd));
    const missingCost = filtered.length - points.length;
    volumeChart.replaceChildren();
    volumeTooltip.hidden = true;
    volumeFrontierCount.textContent = "0";
    volumeCostCount.textContent = String(points.length);
    volumeNote.textContent = "Building the interpolated frontier surface.";
    volumeChart.dataset.surfaceMode = state.surfaceMode;
    surfaceLabel.textContent = {
      dominance: "Dominance-checked triangles", smooth: "Original smooth interpolation",
    }[state.surfaceMode];

    if (!points.length) {
      volumeNote.textContent = "No cost-bearing results match the current filters.";
      volumeChart.append(svgElement("text", {
        class: "volume-empty", x: 540, y: 350, "text-anchor": "middle",
      }, "No cost-bearing results match the current filters."));
      volumeLegend.replaceChildren();
      if (state.view === "frontier3d") visibleCount.textContent = "0";
      return;
    }

    const frontier = paretoVolumePoints(points);
    const frontierIds = new Set(frontier.map((entry) => entry.entry_id));
    volumeFrontierCount.textContent = String(frontier.length);
    if (state.view === "frontier3d") visibleCount.textContent = String(points.length);

    const maximumCost = Math.max(...points.map((entry) => entry.cost_usd));
    const maximumTime = Math.max(...points.map((entry) => entry.time_per_task_sec));
    const costStep = niceStep(Math.max(maximumCost, 0.1) * 1.05 / 5);
    const timeStep = niceStep(Math.max(maximumTime, 1) * 1.08 / 4);
    const costCeiling = Math.max(costStep, Math.ceil(maximumCost * 1.05 / costStep) * costStep);
    const timeCeiling = Math.max(timeStep, Math.ceil(maximumTime * 1.08 / timeStep) * timeStep);
    const minimumPerformance = Math.min(...points.map((entry) => entry.performance));
    const maximumPerformance = Math.max(...points.map((entry) => entry.performance));
    let performanceFloor = Math.max(0, Math.floor(minimumPerformance * 10) / 10);
    let performanceCeiling = Math.min(1, Math.ceil(maximumPerformance * 10) / 10);
    if (performanceCeiling - performanceFloor < 0.2) {
      performanceFloor = Math.max(0, performanceCeiling - 0.2);
    }

    const project = (cost, time, performance) => {
      const centeredCost = cost - 0.5;
      const centeredTime = time - 0.5;
      const cosine = Math.cos(state.volumeYaw);
      const sine = Math.sin(state.volumeYaw);
      const horizontal = centeredCost * cosine - centeredTime * sine;
      const depth = centeredCost * sine + centeredTime * cosine;
      return {
        x: 540 + horizontal * 650,
        y: 520 - performance * 440 * Math.cos(state.volumePitch) +
          depth * 320 * Math.sin(state.volumePitch),
        depth: depth * Math.cos(state.volumePitch) + performance * Math.sin(state.volumePitch),
      };
    };
    const normalized = points.map((entry) => ({
      entry,
      cost: entry.cost_usd / costCeiling,
      time: entry.time_per_task_sec / timeCeiling,
      performance: (entry.performance - performanceFloor) /
        (performanceCeiling - performanceFloor),
      frontier: frontierIds.has(entry.entry_id),
    }));
    const projected = normalized.map((point) => ({
      ...point,
      position: project(point.cost, point.time, point.performance),
      floor: project(point.cost, point.time, 0),
    }));
    const frontierNormalized = normalized.filter((point) => point.frontier);
    const cacheKey = JSON.stringify([state.surfaceMode, frontierNormalized.map((p) =>
      [p.entry.entry_id, p.cost, p.time, p.performance])]);
    let surface = surfaceCache.get(cacheKey);
    if (!surface) {
      surface = state.surfaceMode === "dominance" ? window.ParetoSurface.build(frontierNormalized)
        : paretoSurface(frontierNormalized);
      // Retain a few filter states, never recalculate geometry during rotation.
      if (surfaceCache.size >= 24) surfaceCache.delete(surfaceCache.keys().next().value);
      surfaceCache.set(cacheKey, surface);
    }
    const missingCostNote = missingCost
      ? ` ${missingCost} ${missingCost === 1 ? "result is" : "results are"} omitted because cost is unknown.`
      : "";
    if (state.surfaceMode === "dominance") {
      const rejected = surface.stats.candidates - surface.triangles.length;
      volumeNote.textContent = `${surface.triangles.length} ${surface.triangles.length === 1 ? "triangle" : "triangles"} · ${rejected} candidate patches rejected for dominance. All ${frontier.length} frontier results retained; gaps are intentional.${missingCostNote}`;
    } else {
      volumeNote.textContent = surface.triangles.length
        ? `Original inverse-distance smoothing across ${frontier.length} frontier results; interpolated patches can be dominated.${missingCostNote}`
        : `At least three non-collinear frontier points are needed for the original sheet.${missingCostNote}`;
    }

    const polygon = (coordinates) => coordinates
      .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
    const line = (start, end, className) => volumeChart.append(svgElement("line", {
      class: className,
      x1: start.x.toFixed(1), y1: start.y.toFixed(1),
      x2: end.x.toFixed(1), y2: end.y.toFixed(1),
    }));

    const definitions = svgElement("defs");
    const glow = svgElement("filter", {
      id: "volume-frontier-glow", x: "-100%", y: "-100%", width: "300%", height: "300%",
    });
    glow.append(svgElement("feGaussianBlur", { stdDeviation: 4, result: "blur" }));
    const merge = svgElement("feMerge");
    merge.append(svgElement("feMergeNode", { in: "blur" }));
    merge.append(svgElement("feMergeNode", { in: "SourceGraphic" }));
    glow.append(merge);
    definitions.append(glow);
    volumeChart.append(definitions);

    const floorCorners = [
      project(0, 0, 0), project(1, 0, 0), project(1, 1, 0), project(0, 1, 0),
    ];
    volumeChart.append(svgElement("polygon", {
      class: "volume-floor", points: polygon(floorCorners),
    }));
    for (let index = 0; index <= 4; index += 1) {
      const value = index / 4;
      line(project(value, 0, 0), project(value, 1, 0), "volume-grid");
      line(project(0, value, 0), project(1, value, 0), "volume-grid");
    }

    if (surface.triangles.length) {
      surface.triangles
        .map((triangle) => {
          const projectedTriangle = triangle.map((point) =>
            project(point.cost, point.time, point.performance)
          );
          return {
            points: projectedTriangle,
            vertices: triangle.map((point) => point.entry?.entry_id),
            depth: projectedTriangle.reduce((sum, point) => sum + point.depth, 0) / 3,
            performance: triangle.reduce((sum, point) => sum + point.performance, 0) / 3,
          };
        })
        .sort((left, right) => left.depth - right.depth)
        .forEach((triangle) => {
          const color = surfaceColor(triangle.performance);
          volumeChart.append(svgElement("polygon", {
            class: `pareto-surface-patch${state.surfaceMode === "dominance" ? " dominance-preserving" : ""}`,
            "data-vertices": state.surfaceMode === "dominance"
              ? triangle.vertices.join(",") : "",
            points: polygon(triangle.points),
            fill: color,
            stroke: color,
          }));
        });

    }
    if (state.surfaceMode === "dominance") {
      surface.edges.forEach((edge) => line(
        project(edge[0].cost, edge[0].time, edge[0].performance),
        project(edge[1].cost, edge[1].time, edge[1].performance), "pareto-mesh-edge"
      ));
    }
    if (state.surfaceMode === "smooth" && surface.triangles.length) {
      const boundary = [];
      surface.hull.forEach((point, index) => {
        const next = surface.hull[(index + 1) % surface.hull.length];
        for (let step = 0; step < 10; step += 1) {
          const ratio = step / 10;
          const cost = point.cost + (next.cost - point.cost) * ratio;
          const time = point.time + (next.time - point.time) * ratio;
          boundary.push(project(
            cost,
            time,
            interpolatedPerformance(cost, time, frontierNormalized),
          ));
        }
      });
      const boundaryPath = boundary.map((point, index) =>
        `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`
      ).join(" ");
      volumeChart.append(svgElement("path", {
        class: "pareto-surface-edge", d: `${boundaryPath} Z`,
      }));
    }

    familyNames.forEach((family) => {
      const familyPoints = projected
        .filter((point) => point.entry.model === family)
        .sort((left, right) => left.entry.effort_rank - right.entry.effort_rank ||
          left.entry.performance - right.entry.performance);
      if (familyPoints.length < 2) return;
      const path = familyPoints.map((point, index) =>
        `${index ? "L" : "M"} ${point.position.x.toFixed(1)} ${point.position.y.toFixed(1)}`
      ).join(" ");
      volumeChart.append(svgElement("path", {
        class: "volume-family-path", d: path, stroke: colors.get(family), "data-series": family,
      }));
    });

    const origin = project(0, 0, 0);
    const costEnd = project(1, 0, 0);
    const timeEnd = project(0, 1, 0);
    const performanceEnd = project(0, 0, 1);
    line(origin, costEnd, "volume-axis");
    line(origin, timeEnd, "volume-axis");
    line(origin, performanceEnd, "volume-axis volume-axis-performance");
    [[1, 0], [1, 1], [0, 1]].forEach(([cost, time]) => {
      line(project(cost, time, 0), project(cost, time, 1), "volume-cube-edge");
    });

    for (let index = 0; index <= 4; index += 1) {
      const ratio = index / 4;
      const costPosition = project(ratio, 0, 0);
      const timePosition = project(0, ratio, 0);
      volumeChart.append(svgElement("text", {
        class: "volume-tick volume-cost-tick", x: costPosition.x, y: costPosition.y + 18,
        "text-anchor": "middle",
      }, `$${(costCeiling * ratio).toFixed(costCeiling < 10 ? 2 : 0)}`));
      volumeChart.append(svgElement("text", {
        class: "volume-tick volume-time-tick", x: timePosition.x - 9, y: timePosition.y + 4,
        "text-anchor": "end",
      }, `${Math.round(timeCeiling * ratio)}s`));
    }
    const performanceTickCount = Math.round((performanceCeiling - performanceFloor) * 10);
    for (let index = 0; index <= performanceTickCount; index += 1) {
      const value = performanceFloor + index / 10;
      const ratio = (value - performanceFloor) / (performanceCeiling - performanceFloor);
      const performancePosition = project(0, 0, ratio);
      volumeChart.append(svgElement("text", {
        class: "volume-tick volume-performance-tick",
        x: performancePosition.x - 10, y: performancePosition.y + 4,
        "text-anchor": "end",
      }, `${Math.round(value * 100)}%`));
    }
    volumeChart.append(svgElement("text", {
      class: "volume-axis-label", x: costEnd.x + 12, y: costEnd.y + 30,
      "text-anchor": costEnd.x > 540 ? "start" : "end",
    }, "COST / TASK →"));
    volumeChart.append(svgElement("text", {
      class: "volume-axis-label", x: timeEnd.x - 12, y: timeEnd.y + 24,
      "text-anchor": timeEnd.x > 540 ? "start" : "end",
    }, "TIME / TASK →"));
    volumeChart.append(svgElement("text", {
      class: "volume-axis-label volume-performance-label",
      x: performanceEnd.x, y: performanceEnd.y - 16, "text-anchor": "middle",
    }, "PERFORMANCE ↑"));

    const frontierLabelY = new Map();
    let previousLabelY = -Infinity;
    projected
      .filter((point) => point.frontier)
      .sort((left, right) => left.position.y - right.position.y)
      .forEach((point) => {
        const labelY = Math.max(point.position.y - 10, previousLabelY + 18);
        frontierLabelY.set(point.entry.entry_id, labelY);
        previousLabelY = labelY;
      });

    projected
      .sort((left, right) => Number(left.frontier) - Number(right.frontier) ||
        left.position.depth - right.position.depth)
      .forEach((point, index) => {
        const color = colors.get(point.entry.model);
        line(point.floor, point.position, point.frontier ? "volume-drop frontier" : "volume-drop");
        volumeChart.append(svgElement("circle", {
          class: "volume-footprint", cx: point.floor.x, cy: point.floor.y,
          r: point.frontier ? 4 : 2.5, fill: color,
        }));
        if (point.frontier) {
          volumeChart.append(svgElement("circle", {
            class: "volume-frontier-ring", cx: point.position.x, cy: point.position.y,
            r: 13, stroke: color,
          }));
        }
        const dot = svgElement("circle", {
          class: `volume-point ${point.frontier ? "frontier" : "dominated"}`,
          cx: point.position.x, cy: point.position.y,
          r: point.frontier ? 7 : 4.5,
          fill: color,
          tabindex: 0,
          role: "img",
          "aria-label": `${point.entry.model}, ${point.entry.effort}, ${point.frontier ? "non-dominated" : "dominated"}`,
        });
        dot.style.animationDelay = `${Math.min(index * 22, 280)}ms`;
        dot.addEventListener("pointerenter", (event) => showVolumeTooltip(event, point.entry, point.frontier));
        dot.addEventListener("pointermove", (event) => showVolumeTooltip(event, point.entry, point.frontier));
        dot.addEventListener("pointerleave", () => { volumeTooltip.hidden = true; });
        dot.addEventListener("focus", (event) => showVolumeTooltip(event, point.entry, point.frontier));
        dot.addEventListener("blur", () => { volumeTooltip.hidden = true; });
        volumeChart.append(dot);
        if (point.frontier) {
          const labelOnLeft = point.position.x > 700;
          const labelX = point.position.x + (labelOnLeft ? -17 : 17);
          const labelY = frontierLabelY.get(point.entry.entry_id);
          line(
            point.position,
            { x: labelX + (labelOnLeft ? 5 : -5), y: labelY - 3 },
            "volume-label-leader",
          );
          volumeChart.append(svgElement("text", {
            class: "volume-point-label",
            x: labelX,
            y: labelY,
            fill: color,
            "text-anchor": labelOnLeft ? "end" : "start",
          }, `${point.entry.model} · ${point.entry.effort.replace(" (default)", "")}`));
        }
      });

    const visibleFamilies = familyNames.filter((family) =>
      points.some((entry) => entry.model === family)
    );
    volumeLegend.replaceChildren(...visibleFamilies.map((family) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("i");
      swatch.className = "legend-swatch";
      swatch.style.setProperty("--model-color", colors.get(family));
      item.append(swatch, document.createTextNode(family));
      return item;
    }));
  }

  function render() {
    renderTable();
    renderChart();
    renderVolume();
  }

  root.querySelectorAll("[data-view-button]").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.viewButton;
      root.querySelectorAll("[data-view-button]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-selected", String(active));
      });
      root.querySelectorAll("[data-view-panel]").forEach((panel) => {
        const active = panel.dataset.viewPanel === state.view;
        panel.classList.toggle("active", active);
        panel.hidden = !active;
      });
      render();
    });
  });
  root.querySelectorAll("[data-ranking]").forEach((button) => {
    button.addEventListener("click", () => {
      state.ranking = button.dataset.ranking;
      root.querySelectorAll("[data-ranking]").forEach((candidate) =>
        candidate.classList.toggle("active", candidate === button)
      );
      renderTable();
    });
  });
  root.querySelectorAll("[data-curve]").forEach((button) => {
    button.addEventListener("click", () => {
      state.curve = button.dataset.curve;
      root.querySelectorAll("[data-curve]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      renderChart();
    });
  });
  const cameraPresets = {
    perspective: [-0.72, 0.55],
    "cost-performance": [0, 0.02],
    "time-performance": [-Math.PI / 2, 0.02],
    "cost-time": [-0.72, 1.35],
  };
  root.querySelectorAll("button[data-surface-mode]").forEach((button) => {
    const active = button.dataset.surfaceMode === state.surfaceMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.addEventListener("click", () => {
      state.surfaceMode = button.dataset.surfaceMode;
      root.querySelectorAll("button[data-surface-mode]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      const url = new URL(window.location.href);
      url.searchParams.set("surface", state.surfaceMode);
      url.searchParams.set("view", "frontier3d");
      window.history.replaceState(null, "", url);
      renderVolume();
    });
  });
  root.querySelectorAll("[data-camera-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      [state.volumeYaw, state.volumePitch] = cameraPresets[button.dataset.cameraPreset];
      root.querySelectorAll("[data-camera-preset]").forEach((candidate) =>
        candidate.classList.toggle("active", candidate === button)
      );
      renderVolume();
    });
  });
  let volumeDrag = null;
  volumeFrame?.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest(".volume-point")) return;
    volumeDrag = { x: event.clientX, y: event.clientY };
    volumeFrame.setPointerCapture(event.pointerId);
    volumeFrame.classList.add("dragging");
  });
  volumeFrame?.addEventListener("pointermove", (event) => {
    if (!volumeDrag) return;
    const deltaX = event.clientX - volumeDrag.x;
    const deltaY = event.clientY - volumeDrag.y;
    volumeDrag = { x: event.clientX, y: event.clientY };
    state.volumeYaw += deltaX * 0.008;
    state.volumePitch = Math.max(0.02, Math.min(1.35, state.volumePitch - deltaY * 0.006));
    root.querySelectorAll("[data-camera-preset]").forEach((button) =>
      button.classList.remove("active")
    );
    renderVolume();
    event.preventDefault();
  });
  const endVolumeDrag = (event) => {
    if (!volumeDrag) return;
    volumeDrag = null;
    volumeFrame.classList.remove("dragging");
    if (volumeFrame.hasPointerCapture(event.pointerId)) {
      volumeFrame.releasePointerCapture(event.pointerId);
    }
  };
  volumeFrame?.addEventListener("pointerup", endVolumeDrag);
  volumeFrame?.addEventListener("pointercancel", endVolumeDrag);
  root.querySelectorAll("[data-access]").forEach((button) => {
    button.addEventListener("click", () => {
      state.access = button.dataset.access;
      root.querySelectorAll("[data-access]").forEach((candidate) =>
        candidate.classList.toggle("active", candidate === button)
      );
      render();
    });
  });
  root.addEventListener("change", (event) => {
    const input = event.target;
    if (input.matches("[data-filter-kind=family]")) {
      input.checked ? state.families.add(input.value) : state.families.delete(input.value);
      render();
    } else if (input.matches("[data-filter-kind=effort]")) {
      input.checked ? state.efforts.add(input.value) : state.efforts.delete(input.value);
      render();
    } else if (input.matches("[data-show-all-efforts]")) {
      state.showAllEfforts = input.checked;
      render();
    }
  });
  root.querySelector("[data-reset-filters]").addEventListener("click", () => {
    state.access = "all";
    state.families = new Set(familyNames);
    state.efforts = new Set(effortNames);
    state.showAllEfforts = true;
    root.querySelectorAll("[data-access]").forEach((button) =>
      button.classList.toggle("active", button.dataset.access === "all")
    );
    root.querySelectorAll("[data-filter-kind]").forEach((input) => { input.checked = true; });
    showAllEfforts.checked = true;
    render();
  });

  if (initialParams.get("view") === "frontier3d") {
    root.querySelector('[data-view-button="frontier3d"]').click();
  } else {
    render();
  }
})();
