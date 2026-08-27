// worker.js — POST /adapt : { choose, repair }
// Deploy with: wrangler deploy
// No dependencies, no bindings required.

const CANDIDATE_NAMES = ["prompt_only", "retrieval", "lora", "qlora"];

// ---------- generic helpers ----------

function isPlainObject(x) {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}
function isFiniteNumber(x) {
  return typeof x === "number" && Number.isFinite(x);
}
function isSafeNonNegInt(x) {
  return typeof x === "number" && Number.isSafeInteger(x) && x >= 0;
}
function isPositiveSafeInt(x) {
  return typeof x === "number" && Number.isSafeInteger(x) && x > 0;
}
function round12(x) {
  // avoid binary float noise, round to 12 decimal places
  return Number(x.toFixed(12));
}
function sortDedupe(arr) {
  return Array.from(new Set(arr)).sort(); // ASCII/BMP string sort == UTF-8 byte order here
}

// ---------- operation: choose ----------

function handleChoose(body) {
  const result = { selected: null, eligible: [], totalCosts: {}, reasonCodes: {} };

  const policy = body.policy;
  const policyValid =
    isPlainObject(policy) &&
    isFiniteNumber(policy.minQuality) && policy.minQuality >= 0 && policy.minQuality <= 1 &&
    typeof policy.freshnessRequired === "boolean" &&
    isFiniteNumber(policy.maxLatencyMs) && policy.maxLatencyMs >= 0 &&
    isFiniteNumber(policy.maxMemoryMb) && policy.maxMemoryMb >= 0 &&
    isSafeNonNegInt(policy.maxLabeledExamples) &&
    isFiniteNumber(policy.maxTotalCost) && policy.maxTotalCost >= 0 &&
    isSafeNonNegInt(policy.horizonRequests);

  const candidatesArr = Array.isArray(body.candidates) ? body.candidates : null;

  // Map exactly one candidate per required name. Duplicates / unknown-typed
  // entries / missing array all count as malformed input for the affected name(s).
  const byName = {};
  const seen = new Set();
  let structurallyBroken = candidatesArr === null;

  if (candidatesArr) {
    for (const c of candidatesArr) {
      if (!isPlainObject(c) || typeof c.name !== "string") {
        structurallyBroken = true;
        continue;
      }
      if (!CANDIDATE_NAMES.includes(c.name)) continue; // ignore unrelated entries
      if (seen.has(c.name)) {
        byName[c.name] = undefined; // duplicate -> force invalid for this name
      } else {
        seen.add(c.name);
        byName[c.name] = c;
      }
    }
  }

  for (const name of CANDIDATE_NAMES) {
    const c = byName[name];
    const codes = new Set();
    let totalCost = 0;

    if (!policyValid) codes.add("INVALID_INPUT");

    if (!c) {
      codes.add("INVALID_INPUT");
      totalCost = 0;
    } else {
      const candValid =
        typeof c.available === "boolean" &&
        isFiniteNumber(c.quality) && c.quality >= 0 && c.quality <= 1 &&
        typeof c.freshness === "boolean" &&
        isFiniteNumber(c.latencyMs) && c.latencyMs >= 0 &&
        isFiniteNumber(c.memoryMb) && c.memoryMb >= 0 &&
        isSafeNonNegInt(c.labeledExamples) &&
        isFiniteNumber(c.oneTimeCost) && c.oneTimeCost >= 0 &&
        isFiniteNumber(c.recurringCost) && c.recurringCost >= 0;

      if (!candValid) {
        codes.add("INVALID_INPUT");
        if (isFiniteNumber(c.oneTimeCost) && isFiniteNumber(c.recurringCost) && policyValid) {
          totalCost = round12(c.oneTimeCost + policy.horizonRequests * c.recurringCost);
        }
      } else if (policyValid) {
        totalCost = round12(c.oneTimeCost + policy.horizonRequests * c.recurringCost);
        if (!c.available) codes.add("UNAVAILABLE");
        if (c.quality < policy.minQuality) codes.add("QUALITY_FLOOR");
        if (policy.freshnessRequired && !c.freshness) codes.add("FRESHNESS_REQUIRED");
        if (c.latencyMs > policy.maxLatencyMs) codes.add("LATENCY_LIMIT");
        if (c.memoryMb > policy.maxMemoryMb) codes.add("MEMORY_LIMIT");
        if (c.labeledExamples > policy.maxLabeledExamples) codes.add("DATA_LIMIT");
        if (totalCost > policy.maxTotalCost) codes.add("COST_LIMIT");
      } else {
        totalCost = round12(c.oneTimeCost + 0 * c.recurringCost);
      }
    }

    result.totalCosts[name] = totalCost;
    result.reasonCodes[name] = sortDedupe(Array.from(codes));
    if (codes.size === 0) result.eligible.push(name);
  }

  result.selected = result.eligible.length ? result.eligible[0] : null;
  return result;
}

// ---------- operation: repair ----------

function validateToken(t) {
  return (
    isPlainObject(t) &&
    isSafeNonNegInt(t.id) &&
    ["system", "user", "assistant"].includes(t.role) &&
    typeof t.padding === "boolean" &&
    typeof t.text === "string"
  );
}

function validIdArray(arr) {
  return (
    Array.isArray(arr) &&
    arr.length > 0 &&
    arr.every((x) => typeof x === "string" && x.length > 0) &&
    new Set(arr).size === arr.length
  );
}

function validNumArray(arr) {
  return Array.isArray(arr) && arr.length > 0 && arr.every((x) => typeof x === "number" && Number.isFinite(x));
}

function handleRepair(body) {
  const reasonCodes = new Set();

  // --- tokens / labels ---
  const tokens = Array.isArray(body.tokens) ? body.tokens : [];
  const tokensNonEmpty = tokens.length > 0;
  const allTokensValid = tokensNonEmpty && tokens.every(validateToken);
  let labels;
  if (!allTokensValid) {
    reasonCodes.add("INVALID_TOKEN");
    labels = tokens.map(() => -100);
  } else {
    labels = tokens.map((t) => (t.role === "assistant" && t.padding === false ? t.id : -100));
  }

  // --- chat template ---
  const templatePass = body.templateApplications === 1;
  if (!templatePass) reasonCodes.add("CHAT_TEMPLATE_COUNT");

  // --- PEFT parameters ---
  let peftConfigPass = true;
  let trainableParams = [];
  let trainableCount = 0;

  const parameters = Array.isArray(body.parameters) ? body.parameters : null;
  const allowedTargets = Array.isArray(body.allowedTargets) ? body.allowedTargets : null;

  const allowedTargetsValid =
    !!allowedTargets &&
    allowedTargets.length > 0 &&
    allowedTargets.every((t) => typeof t === "string") &&
    new Set(allowedTargets).size === allowedTargets.length;

  let paramsValid = !!parameters && parameters.length > 0;
  if (paramsValid) {
    const names = new Set();
    for (const p of parameters) {
      if (
        !isPlainObject(p) ||
        typeof p.name !== "string" ||
        p.name.length === 0 ||
        typeof p.target !== "string" ||
        !isPositiveSafeInt(p.numel)
      ) {
        paramsValid = false;
        break;
      }
      if (names.has(p.name)) {
        paramsValid = false;
        break;
      }
      names.add(p.name);
    }
  }

  if (!allowedTargetsValid || !paramsValid) {
    peftConfigPass = false;
    reasonCodes.add("INVALID_PARAMETER");
  } else {
    const allowedSet = new Set(allowedTargets);
    const qualifying = parameters.filter(
      (p) => allowedSet.has(p.target) && (p.name.endsWith(".lora_A.weight") || p.name.endsWith(".lora_B.weight"))
    );
    if (qualifying.length === 0) {
      peftConfigPass = false;
      reasonCodes.add("INVALID_PARAMETER");
    } else {
      trainableParams = qualifying.map((p) => p.name).sort();
      trainableCount = qualifying.reduce((s, p) => s + p.numel, 0);
    }
  }

  if (body.inferenceMode !== false) {
    peftConfigPass = false;
    reasonCodes.add("INFERENCE_MODE");
  }

  // --- artifact files ---
  const requiredFiles = ["adapter_config.json", "adapter_model.safetensors"];
  const artifactFiles = Array.isArray(body.artifactFiles) ? body.artifactFiles : [];
  const sortedActual = [...artifactFiles].sort();
  const sortedRequired = [...requiredFiles].sort();
  const setMatches =
    sortedActual.length === sortedRequired.length && sortedActual.every((f, i) => f === sortedRequired[i]);

  let adapterFiles = [];
  if (setMatches) {
    adapterFiles = sortedRequired;
  } else {
    reasonCodes.add("ADAPTER_FILE_SET");
    const fullModelIndicators = new Set([
      "pytorch_model.bin",
      "model.safetensors",
      "model.bin",
      "pytorch_model.safetensors",
    ]);
    if (artifactFiles.some((f) => fullModelIndicators.has(f))) {
      reasonCodes.add("FULL_MODEL_ARTIFACT");
    }
  }

  // --- checkpoint ---
  const requiredCheckpointKeys = ["model", "optimizer", "scheduler", "step", "rng", "dataPosition"];
  const checkpoint = isPlainObject(body.checkpoint) ? body.checkpoint : {};
  const checkpointComplete = requiredCheckpointKeys.every((k) =>
    Object.prototype.hasOwnProperty.call(checkpoint, k)
  );
  if (!checkpointComplete) reasonCodes.add("INCOMPLETE_CHECKPOINT");

  // --- lineage ---
  const hex40 = /^[0-9a-f]{40}$/;
  const hex64 = /^[0-9a-f]{64}$/;
  let lineagePass = true;

  const expectedDigests = isPlainObject(body.expectedDigests) ? body.expectedDigests : {};
  const baseRevisionFormatValid = typeof body.baseRevision === "string" && hex40.test(body.baseRevision);

  if (!baseRevisionFormatValid) {
    lineagePass = false;
    reasonCodes.add("MUTABLE_BASE_REVISION");
  } else if (expectedDigests.baseRevision !== undefined && expectedDigests.baseRevision !== body.baseRevision) {
    lineagePass = false;
    reasonCodes.add("MUTABLE_BASE_REVISION");
  }

  let digestProblem = false;
  for (const field of ["datasetDigest", "codeDigest", "configDigest"]) {
    const val = body[field];
    if (typeof val !== "string" || !hex64.test(val)) {
      digestProblem = true;
      continue;
    }
    if (expectedDigests[field] !== undefined && expectedDigests[field] !== val) {
      digestProblem = true;
    }
  }
  if (digestProblem) {
    lineagePass = false;
    reasonCodes.add("LINEAGE_MISMATCH");
  }

  // --- effective batch ---
  const mb = body.microBatch,
    ga = body.gradientAccumulation,
    rep = body.replicas,
    eb = body.expectedEffectiveBatch;
  const batchFieldsValid = [mb, ga, rep, eb].every(isPositiveSafeInt);
  if (!batchFieldsValid || mb * ga * rep !== eb) {
    reasonCodes.add("EFFECTIVE_BATCH_MISMATCH");
  }

  // --- eval isolation ---
  let evalIsolated = true;
  const trainValid = validIdArray(body.trainRowIds);
  const evalValid = validIdArray(body.evalRowIds);
  if (!trainValid || !evalValid) {
    evalIsolated = false;
    reasonCodes.add("EVAL_LEAKAGE");
  } else {
    const trainSet = new Set(body.trainRowIds);
    if (body.evalRowIds.some((id) => trainSet.has(id))) {
      evalIsolated = false;
      reasonCodes.add("EVAL_LEAKAGE");
    }
  }

  // --- eval determinism ---
  let evaluationDeterministic = true;
  if (body.dropoutActiveDuringEval !== false) {
    evaluationDeterministic = false;
    reasonCodes.add("EVAL_DROPOUT_ACTIVE");
  }

  // --- resume ---
  let resumePass = true;
  const uw = body.uninterruptedWeights,
    rw = body.resumedWeights,
    tol = body.resumeTolerance;
  const uwValid = validNumArray(uw);
  const rwValid = validNumArray(rw);
  const tolValid = isFiniteNumber(tol) && tol >= 0;

  if (!uwValid || !rwValid || !tolValid || uw.length !== rw.length) {
    resumePass = false;
    reasonCodes.add("RESUME_DIVERGENCE");
  } else {
    for (let i = 0; i < uw.length; i++) {
      if (Math.abs(uw[i] - rw[i]) > tol) {
        resumePass = false;
        reasonCodes.add("RESUME_DIVERGENCE");
        break;
      }
    }
  }

  return {
    labels,
    templatePass,
    trainableParams,
    trainableCount,
    peftConfigPass,
    adapterFiles,
    checkpointComplete,
    lineagePass,
    evalIsolated,
    evaluationDeterministic,
    resumePass,
    reasonCodes: sortDedupe(Array.from(reasonCodes)),
  };
}

module.exports = { isPlainObject, handleChoose, handleRepair };
