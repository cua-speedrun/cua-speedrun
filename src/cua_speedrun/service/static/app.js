/* Live run page: subscribes to the run's SSE stream and updates the stage
   timeline, task grid, and log pane in place. No framework, no build step:
   the server renders truth, this file only keeps it current. */

(function () {
  const root = document.querySelector("[data-run-id]");
  if (!root) return;
  const runId = root.dataset.runId;
  const terminal = ["card_ready", "failed", "rejected", "held", "cancelled"];
  const stopButton = document.querySelector("[data-stop-run]");
  const resumeButton = document.querySelector("[data-resume-run]");

  if (resumeButton) {
    resumeButton.addEventListener("click", async () => {
      if (!window.confirm(
        `Resume evaluation ${runId}? Completed task results will be kept.`
      )) return;
      resumeButton.disabled = true;
      resumeButton.textContent = "Resuming…";
      try {
        const response = await fetch(`/api/runs/${runId}/resume`, {
          method: "POST",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `Resume failed (${response.status})`);
        }
        window.location.reload();
      } catch (error) {
        resumeButton.disabled = false;
        resumeButton.textContent = "Resume unfinished tasks";
        window.alert(error.message);
      }
    });
  }

  if (stopButton && !stopButton.disabled) {
    stopButton.addEventListener("click", async () => {
      if (!window.confirm(`Stop evaluation ${runId}? Partial logs will be kept.`)) return;
      stopButton.disabled = true;
      stopButton.textContent = "Stopping…";
      try {
        const response = await fetch(`/api/runs/${runId}/cancel`, {
          method: "POST",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `Stop failed (${response.status})`);
        }
        const result = await response.json();
        if (result.stage === "cancelled") window.location.reload();
      } catch (error) {
        stopButton.disabled = false;
        stopButton.textContent = "Stop evaluation";
        window.alert(error.message);
      }
    });
  }

  // A finished run is a static page: opening the live stream would just
  // receive the terminal stage and reload, then reconnect and reload again
  // in a loop (the flashing). Only stream while the run is in progress.
  if (terminal.includes(root.dataset.stage)) return;

  const logPane = document.getElementById("log");
  const stageEls = Array.from(document.querySelectorAll(".stage"));
  const STAGES = stageEls.map((el) => el.dataset.stage);

  function setStage(stage) {
    const idx = STAGES.indexOf(stage);
    stageEls.forEach((el, i) => {
      el.classList.toggle("done", idx >= 0 && i < idx);
      el.classList.toggle("now", i === idx);
      if (i === idx && stage === "card_ready") el.classList.add("terminal-ok");
      if (i === idx && ["failed", "rejected", "held", "cancelled"].includes(stage))
        el.classList.add("terminal-bad");
    });
    if (terminal.includes(stage) && !reloaded) {
      // A live run just finished: re-render once from the server for the final
      // card and numbers. Guarded so repeated stage events can't stack reloads;
      // after the reload the page is terminal and never opens the stream again.
      reloaded = true;
      es.close();
      setTimeout(() => window.location.reload(), 800);
    }
  }
  let reloaded = false;

  function taskCell(key) {
    let cell = document.querySelector(`.task-cell[data-key="${CSS.escape(key)}"]`);
    if (cell) return cell;
    const empty = document.getElementById("tasks-empty");
    if (empty) empty.remove();
    cell = document.createElement("div");
    cell.className = "task-cell";
    cell.dataset.key = key;
    cell.innerHTML =
      '<div class="tk"></div><div class="st"><span class="chip muted stage-chip"></span><span class="t"></span></div>';
    cell.querySelector(".tk").textContent = key;
    document.querySelector(".task-grid").appendChild(cell);
    return cell;
  }

  const TASK_STAGE = {
    task_started: "env boot",
    env_created: "env boot",
    warmup_started: "warming",
    warmup_fallback: "warming",
    env_ready: "env ready",
    warmup_done: "warm",
    armed: "running",
    agent_exit: "checking",
    task_done: "done",
    task_failed: "failed",
  };

  function logLine(text, kind) {
    if (!logPane) return;
    const atBottom =
      logPane.scrollTop + logPane.clientHeight >= logPane.scrollHeight - 8;
    const line = document.createElement("div");
    if (kind) {
      const k = document.createElement("span");
      k.className = "k";
      k.textContent = kind + "  ";
      line.appendChild(k);
    }
    line.appendChild(document.createTextNode(text));
    logPane.appendChild(line);
    if (atBottom) logPane.scrollTop = logPane.scrollHeight;
  }

  function onEvent(ev) {
    if (ev.kind === "init_line") {
      logLine(ev.line.replace(/\n$/, ""));
      return;
    }
    const label = TASK_STAGE[ev.kind];
    if (ev.task && label) {
      const cell = taskCell(ev.task);
      cell.dataset.stage = ev.kind === "task_done" ? "done" : label.replace(" ", "_");
      const chip = cell.querySelector(".stage-chip");
      chip.textContent = label;
      chip.className =
        "chip stage-chip " + (label === "running" ? "live" : "muted");
      if (ev.kind === "task_done") {
        cell.dataset.passed = String(!!ev.passed);
        chip.textContent = ev.passed ? "pass" : "fail";
        chip.className = "chip stage-chip " + (ev.passed ? "pass" : "fail");
        if (ev.task_time_sec != null)
          cell.querySelector(".t").textContent = ev.task_time_sec.toFixed(2) + "s";
        if (ev.cost_usd != null) {
          let cost = cell.querySelector(".cost");
          if (!cost) {
            cost = document.createElement("span");
            cost.className = "cost";
            cell.querySelector(".t").parentElement.append(" · ", cost);
          }
          cost.textContent = "$" + ev.cost_usd.toFixed(4);
        }
      }
    }
    logLine(JSON.stringify(ev), ev.kind);
  }

  // Resume after the server-rendered history so nothing shows twice.
  const lastId = parseInt(root.dataset.lastEventId || "0", 10);
  const es = new EventSource(`/api/runs/${runId}/stream?after=${lastId}`);
  es.addEventListener("run", (m) => onEvent(JSON.parse(m.data)));
  es.addEventListener("stage", (m) => setStage(JSON.parse(m.data).stage));
  es.addEventListener("end", () => es.close());
  es.onerror = () => {
    /* the browser retries automatically; nothing to do */
  };
})();

/* New evaluation: first freeze the maintainer-owned contract, then attach
   either a starter or an uploaded two-file agent. */
(function () {
  const form = document.getElementById("submit-form");
  if (!form) return;

  const panels = Array.from(form.querySelectorAll("[data-wizard-panel]"));
  const navItems = Array.from(form.querySelectorAll("[data-wizard-nav]"));
  const trackInputs = Array.from(form.querySelectorAll("[name=track]"));
  const computeInputs = Array.from(form.querySelectorAll("[name=compute_placement]"));
  const environmentInputs = Array.from(form.querySelectorAll("[name=environment_placement]"));
  const executionTopologies = JSON.parse(form.dataset.executionTopologies || "[]");
  const benchmarkSelect = form.querySelector("[name=benchmark_id]");
  const templateRadios = Array.from(form.querySelectorAll("[name=template_choice]"));
  const templateCards = Array.from(form.querySelectorAll("[data-template-card]"));
  const savedEnvironmentChecks = Array.from(
    form.querySelectorAll("[name=saved_environment_choice]")
  );
  const evaluationEnvironmentList = form.querySelector("[data-evaluation-environment-list]");
  const evaluationEnvironmentEmpty = form.querySelector("[data-evaluation-environment-empty]");
  const evaluationEnvironmentTemplate = document.getElementById("evaluation-environment-row");
  const fileInput = form.querySelector("input[type=file]");
  const nameInput = form.querySelector("[name=name]");
  const errorBox = document.getElementById("submit-error");
  const startButton = form.querySelector("[data-start-evaluation]");
  const nextButton = form.querySelector("[data-wizard-next]");
  const placementStatus = form.querySelector("[data-placement-status]");
  const gpuInput = form.querySelector("[name=gpu]");
  const algorithmSelect = form.querySelector("[name=eval_algorithm]");
  const parallelEvaluations = form.querySelector("[name=parallel_evaluations]");
  const parallelTitle = form.querySelector("[data-parallel-title]");
  const parallelDescription = form.querySelector("[data-parallel-description]");
  const parallelControl = form.querySelector("[data-parallel-evaluations]");
  let agentSource = "starter";
  gpuInput.value = selectedTrack()?.dataset.trackGpu || "";
  algorithmSelect.value = selectedTrack()?.dataset.trackAlgorithm || "";
  let parallelPreference = Number.parseInt(parallelEvaluations.value || "1", 10);

  function selectedTrack() {
    return form.querySelector("[name=track]:checked");
  }

  function selectedBenchmark() {
    return benchmarkSelect.options[benchmarkSelect.selectedIndex];
  }

  function selectedCompute() {
    return form.querySelector("[name=compute_placement]:checked");
  }

  function selectedEnvironment() {
    return form.querySelector("[name=environment_placement]:checked");
  }

  function agentsPerEvaluation() {
    const track = selectedTrack();
    return Number.parseInt(track?.dataset.trackAgentsPerEvaluation || "1", 10);
  }

  function parallelEvaluationCount() {
    const value = Number.parseInt(parallelEvaluations.value || "1", 10);
    return Number.isFinite(value) ? value : 1;
  }

  function refreshParallelism() {
    const supported = algorithmSelect.selectedOptions[0]?.dataset.parallelEvaluations === "true";
    parallelEvaluations.disabled = !supported;
    parallelControl.dataset.state = supported ? "available" : "unavailable";
    if (!supported) {
      parallelEvaluations.value = "1";
      parallelTitle.textContent = "Single evaluation";
      parallelDescription.textContent =
        "The fresh-sandbox algorithm runs one evaluation at a time.";
      return;
    }
    parallelEvaluations.value = String(parallelPreference);
    const parallel = parallelEvaluationCount();
    const agents = agentsPerEvaluation();
    const totalAgents = parallel * agents;
    parallelTitle.textContent = `${parallel} parallel evaluation${parallel === 1 ? "" : "s"}`;
    parallelDescription.textContent =
      `${agents} agent${agents === 1 ? "" : "s"} per isolated evaluation · ` +
      `${totalAgents} active agent${totalAgents === 1 ? "" : "s"} and environment VM${totalAgents === 1 ? "" : "s"}.`;
  }

  function selectedTopology() {
    const compute = selectedCompute();
    const environment = selectedEnvironment();
    if (!compute || !environment) return null;
    return executionTopologies.find((topology) =>
      topology.compute.key === compute.value &&
      topology.environment.key === environment.value
    ) || null;
  }

  function placementIssue() {
    const track = selectedTrack();
    if (!track || !selectedCompute() || !selectedEnvironment()) {
      return "Choose both execution placements.";
    }
    const topology = selectedTopology();
    if (!topology) {
      return "This model/agent and environment-VM combination does not have a registered executor yet.";
    }
    if (!topology.eval_algorithms.includes(algorithmSelect.value)) {
      const placement = `${selectedCompute().dataset.placementLabel} model/agent + ${selectedEnvironment().dataset.placementLabel} VMs`;
      return `${placement} does not support ${algorithmSelect.selectedOptions[0]?.textContent}. Choose another execution mode.`;
    }
    return "";
  }

  function refreshPlacementStatus() {
    const issue = placementIssue();
    placementStatus.textContent = issue || "Available for this track.";
    placementStatus.dataset.state = issue ? "error" : "ready";
    nextButton.disabled = Boolean(issue);
    refreshContractSummary();
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
    errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function clearError() {
    errorBox.style.display = "none";
    errorBox.textContent = "";
  }

  function showStep(step) {
    panels.forEach((panel) => { panel.hidden = panel.dataset.wizardPanel !== step; });
    navItems.forEach((item) => {
      const active = item.dataset.wizardNav === step;
      item.classList.toggle("active", active);
      item.classList.toggle("complete", step === "agent" && item.dataset.wizardNav === "contract");
      if (active) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
    if (step === "agent") {
      navItems.find((item) => item.dataset.wizardNav === "agent").disabled = false;
      refreshContractSummary();
      refreshCompatibleTemplates();
    }
    window.scrollTo(0, 0);
  }

  function refreshContractSummary() {
    const track = selectedTrack();
    const benchmark = selectedBenchmark();
    const compute = selectedCompute();
    const environment = selectedEnvironment();
    if (!track || !benchmark || !compute || !environment) return;
    form.querySelector("[data-summary-track]").textContent = track.dataset.trackLabel;
    form.querySelector("[data-summary-benchmark]").textContent = benchmark.dataset.benchmarkLabel;
    form.querySelector("[data-summary-hardware]").textContent =
      gpuInput.value.trim()
        ? `${parallelEvaluationCount()} × ${gpuInput.value.trim()}`
        : "None";
    form.querySelector("[data-summary-algorithm]").textContent =
      algorithmSelect.selectedOptions[0]?.textContent || "—";
    form.querySelector("[data-summary-parallel]").textContent = parallelEvaluationCount();
    form.querySelector("[data-summary-agents]").textContent =
      parallelEvaluationCount() * agentsPerEvaluation();
    form.querySelector("[data-summary-compute]").textContent = compute.dataset.placementLabel;
    form.querySelector("[data-summary-environment]").textContent = environment.dataset.placementLabel;
    form.querySelector("[data-summary-success]").textContent = track.dataset.trackSuccess;
  }

  function refreshCompatibleTemplates() {
    const benchmark = selectedBenchmark();
    if (!benchmark) return;
    let visible = 0;
    templateCards.forEach((card) => {
      const compatibleBenchmarks = (card.dataset.compatibleBenchmarks || "").split(",");
      const compatible = compatibleBenchmarks.includes(benchmark.dataset.benchmarkName);
      card.hidden = !compatible;
      if (compatible) visible += 1;
      const radio = card.querySelector("[name=template_choice]");
      if (!compatible && radio.checked) radio.checked = false;
    });
    form.querySelector("[data-no-compatible-template]").hidden = visible !== 0;
  }

  function setAgentSource(source) {
    agentSource = source;
    form.querySelectorAll("[data-agent-source]").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.agentSource === source));
    });
    form.querySelectorAll("[data-agent-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.agentPanel !== source;
    });
    fileInput.required = source === "upload";
    clearError();
  }

  function refreshEvaluationEnvironmentEmpty() {
    evaluationEnvironmentEmpty.hidden = Boolean(evaluationEnvironmentList.children.length);
  }

  function addEvaluationVariable(name = "", origin = "user") {
    const fragment = evaluationEnvironmentTemplate.content.cloneNode(true);
    const row = fragment.querySelector(".evaluation-environment-row");
    const nameField = row.querySelector("[data-environment-name]");
    row.dataset.origin = origin;
    nameField.value = name;
    row.querySelector("[data-remove-environment-variable]").addEventListener("click", () => {
      row.remove();
      refreshEvaluationEnvironmentEmpty();
    });
    evaluationEnvironmentList.appendChild(fragment);
    refreshEvaluationEnvironmentEmpty();
    if (!name) nameField.focus();
    return row;
  }

  function ensureEvaluationVariable(name) {
    const existing = Array.from(
      evaluationEnvironmentList.querySelectorAll(".evaluation-environment-row")
    ).find((row) => row.querySelector("[data-environment-name]").value.trim() === name);
    return existing || addEvaluationVariable(name, "template");
  }

  function readEvaluationVariables() {
    const variables = [];
    const seen = new Set();
    const rows = Array.from(
      evaluationEnvironmentList.querySelectorAll(".evaluation-environment-row")
    );
    for (const row of rows) {
      const nameField = row.querySelector("[data-environment-name]");
      const valueField = row.querySelector("[data-environment-value]");
      const name = nameField.value.trim();
      const value = valueField.value;
      nameField.setCustomValidity("");
      valueField.setCustomValidity("");
      if (!name && !value) continue;
      if (!name) {
        nameField.setCustomValidity("Enter a variable name.");
        nameField.reportValidity();
        return null;
      }
      if (!nameField.reportValidity()) return null;
      if (!value) {
        valueField.setCustomValidity("Enter a value.");
        valueField.reportValidity();
        return null;
      }
      if (seen.has(name)) {
        nameField.setCustomValidity(`${name} is already listed for this evaluation.`);
        nameField.reportValidity();
        return null;
      }
      seen.add(name);
      variables.push({ name, value });
    }
    return variables;
  }

  function requiredEnvironment(card) {
    return (card.dataset.requiredEnvironment || "").split(",").filter(Boolean);
  }

  async function submit(body) {
    clearError();
    let resp;
    let data;
    try {
      resp = await fetch("/api/submissions", { method: "POST", body });
      data = await resp.json().catch(() => ({}));
    } catch (_error) {
      showError("The evaluation service could not be reached. Try again.");
      return false;
    }
    if (!resp.ok) {
      showError(data.detail || "Submission failed.");
      return false;
    }
    window.location.href = `/runs/${data.run_id}`;
    return true;
  }

  nextButton.addEventListener("click", () => {
    if (!selectedTrack() || !benchmarkSelect.value || placementIssue()) return;
    showStep("agent");
  });

  form.querySelectorAll("[data-wizard-back]").forEach((button) => {
    button.addEventListener("click", () => showStep("contract"));
  });

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      if (!item.disabled) showStep(item.dataset.wizardNav);
    });
  });

  form.querySelectorAll("[data-agent-source]").forEach((button) => {
    button.addEventListener("click", () => setAgentSource(button.dataset.agentSource));
  });

  trackInputs.forEach((input) => input.addEventListener("change", () => {
    gpuInput.value = selectedTrack()?.dataset.trackGpu || "";
    algorithmSelect.value = selectedTrack()?.dataset.trackAlgorithm || "";
    refreshParallelism();
    refreshPlacementStatus();
    refreshCompatibleTemplates();
  }));
  gpuInput.addEventListener("input", refreshContractSummary);
  algorithmSelect.addEventListener("change", () => {
    refreshParallelism();
    refreshPlacementStatus();
  });
  parallelEvaluations.addEventListener("input", () => {
    parallelPreference = parallelEvaluationCount();
    refreshParallelism();
    refreshContractSummary();
    refreshPlacementStatus();
  });
  [...computeInputs, ...environmentInputs].forEach((input) => {
    input.addEventListener("change", refreshPlacementStatus);
  });
  benchmarkSelect.addEventListener("change", refreshCompatibleTemplates);
  form.querySelector("[data-add-environment-variable]").addEventListener(
    "click", () => addEvaluationVariable()
  );
  templateRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (!nameInput.value.trim()) nameInput.value = radio.value;
      const required = requiredEnvironment(radio.closest("[data-template-card]"));
      Array.from(
        evaluationEnvironmentList.querySelectorAll('[data-origin="template"]')
      ).forEach((row) => {
        if (!row.querySelector("[data-environment-value]").value) row.remove();
      });
      refreshEvaluationEnvironmentEmpty();
      required.forEach((name) => {
        const saved = savedEnvironmentChecks.find((check) => check.value === name);
        if (saved) saved.checked = true;
        else ensureEvaluationVariable(name);
      });
      clearError();
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    if (!nameInput.reportValidity()) return;
    if (!parallelEvaluations.reportValidity()) return;
    const track = selectedTrack();
    const compute = selectedCompute();
    const environment = selectedEnvironment();
    if (!track || !benchmarkSelect.value || !compute || !environment || placementIssue()) {
      showStep("contract");
      refreshPlacementStatus();
      return;
    }

    const evaluationVariables = readEvaluationVariables();
    if (evaluationVariables === null) return;
    const selectedSavedVariables = savedEnvironmentChecks
      .filter((item) => item.checked).map((item) => item.value);
    const body = new FormData();
    body.append("name", nameInput.value.trim());
    body.append("track", track.value);
    body.append("benchmark_id", benchmarkSelect.value);
    body.append("compute_placement", compute.value);
    body.append("environment_placement", environment.value);
    body.append("allocate_gpu", String(Boolean(gpuInput.value.trim())));
    if (gpuInput.value.trim()) body.append("gpu", gpuInput.value.trim());
    body.append("eval_algorithm", algorithmSelect.value);
    body.append("parallel_evaluations", String(parallelEvaluationCount()));
    body.append("saved_environment_variables", JSON.stringify(selectedSavedVariables));
    body.append("evaluation_environment_variables", JSON.stringify(evaluationVariables));
    if (agentSource === "starter") {
      const template = form.querySelector("[name=template_choice]:checked");
      if (!template || template.closest("[data-template-card]").hidden) {
        showError("Select a starter agent or upload your own ZIP.");
        return;
      }
      const required = requiredEnvironment(template.closest("[data-template-card]"));
      const providedEnvironmentNames = new Set(
        selectedSavedVariables.concat(evaluationVariables.map((item) => item.name))
      );
      const missingVariables = required.filter(
        (name) => !providedEnvironmentNames.has(name)
      );
      if (missingVariables.length) {
        showError(`This starter requires ${missingVariables.join(", ")}. Select a saved variable or add it for this evaluation.`);
        return;
      }
      body.append("template", template.value);
    } else {
      if (!fileInput.files.length) {
        fileInput.reportValidity();
        return;
      }
      body.append("file", fileInput.files[0]);
    }

    const idleLabel = startButton.textContent;
    startButton.disabled = true;
    startButton.textContent = "Starting…";
    const started = await submit(body);
    if (!started) {
      startButton.disabled = false;
      startButton.textContent = idleLabel;
    }
  });

  refreshParallelism();
  refreshPlacementStatus();
  refreshCompatibleTemplates();
  setAgentSource("starter");
})();

/* Publish form on the private card. */
(function () {
  const form = document.getElementById("publish-form");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = document.getElementById("publish-error");
    err.style.display = "none";
    const name = encodeURIComponent(form.querySelector("[name=entry_name]").value);
    const resp = await fetch(
      `/api/cards/${form.dataset.token}/publish?entry_name=${name}`,
      { method: "POST" }
    );
    const body = await resp.json();
    if (!resp.ok) {
      err.textContent = body.detail || "publish failed";
      err.style.display = "block";
      return;
    }
    window.location.href = `/?season=${encodeURIComponent(body.season_key)}`;
  });
})();

/* Exact-season entry picker on the Frontier. The server repeats every
   boundary check; this only makes the 2–4 point interaction immediate. */
(function () {
  const form = document.getElementById("compare-picker");
  if (!form) return;
  const checks = Array.from(form.querySelectorAll(".compare-check"));
  const count = form.querySelector("[data-compare-count]");
  const status = form.querySelector("[data-compare-status]");
  const submit = form.querySelector("[data-compare-submit]");
  const dock = form.querySelector("[data-compare-dock]");

  function update() {
    const selected = checks.filter((check) => check.checked).length;
    count.textContent = String(selected);
    checks.forEach((check) => {
      check.disabled = selected >= 4 && !check.checked;
    });
    submit.disabled = selected < 2;
    submit.textContent = selected >= 2 ? `Compare ${selected} entries` : "Compare selected";
    status.textContent =
      selected === 0 ? "Select two to four points" :
      selected === 1 ? "Select one more point" :
      selected === 4 ? "Maximum selection reached" :
      "Ready for direct comparison";
    dock.classList.toggle("ready", selected >= 2);
  }

  checks.forEach((check) => check.addEventListener("change", update));
  update();
})();

/* Client-side filter for the already-rendered evaluation history. */
(function () {
  const input = document.getElementById("evaluation-search");
  const list = document.getElementById("evaluation-list");
  if (!input || !list) return;
  const rows = Array.from(list.querySelectorAll(".evaluation-row:not(.evaluation-header)"));
  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase();
    rows.forEach((row) => {
      row.hidden = query && !row.dataset.search.includes(query);
    });
  });
})();

/* Load complete agent streams only when someone opens them. */
(function () {
  document.querySelectorAll("[data-agent-output-url]").forEach((panel) => {
    panel.addEventListener("toggle", async () => {
      if (!panel.open || panel.dataset.loaded === "true") return;
      const output = panel.querySelector("[data-agent-output]");
      output.textContent = "Loading…";
      try {
        const response = await fetch(panel.dataset.agentOutputUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        output.textContent = await response.text() || "No output was captured.";
        panel.dataset.loaded = "true";
      } catch (error) {
        output.textContent = `Could not load agent output: ${error.message}`;
      }
    });
  });
})();
