# Prompt Excerpts

This file summarizes the main reproducibility-relevant instructions from the four
pipeline prompts in `prompts/`: extractor, auditor, code generation, and code
auditor.

## 1. Extractor Prompt

Source: `prompts/extractor.yaml`

Purpose:

> You are a scientific computing assistant specialized in energy communities,
> electricity markets, flexibility services, energy storage, and business model
> simulation.

Core task:

> Your task is NOT to summarize the paper. Your task is to reconstruct a complete
> executable computational model from the paper. The final template will later be
> converted automatically into Python code. Therefore completeness is more
> important than brevity.

Dataset grounding:

> Every energy community dataset has EXACTLY these columns:
> `Timestamp`, `member_id`, `member_type`, `load`, `generation`.

Input binding rule:

> Every variable extracted from the paper that depends on time-series data MUST
> ultimately be mapped to these columns or to variables derived from them.
> Do NOT invent new input columns: if a paper quantity is time-dependent but is
> not one of the five canonical columns, it is a DERIVED variable, not an input.

Tariff binding rule:

> A paper tariff parameter that matches a tariff CSV column MUST be represented
> in `parameters[]` with `value_source="ec_tariff"`, `tariff_key` set to the
> exact case-sensitive CSV column name, and `value=null`.

Extraction strategy:

> Do NOT start from equations. Instead identify every symbol, variable, metric,
> parameter, coefficient, index, state variable, decision variable, intermediate
> result, and final output appearing in equations, algorithms, pseudocode,
> methodology sections, tables, figures, examples, and case studies.

Required symbol typing:

> Infer the shape of every symbol and record it in the `type` field using exactly
> one of: `scalar`, `vector`, `matrix`, `array`, `string`, `datetime`.

Equation extraction:

> Never omit equations because they are not numbered. If an equation is referenced
> indirectly by an algorithm, extract it. For every equation provide: equation id,
> expression, output variable, dependencies, and source reference.

Workflow reconstruction:

> Reconstruct the complete computational workflow: every equation must appear in
> at least one step; every derived variable must be computed exactly once; every
> output must be produced; every dependency must be available before use.

Validation before return:

> Before generating JSON verify that every variable in equations and steps exists
> in symbols, every dependency references an existing symbol, every output appears
> in steps, every equation output appears in symbols, every used parameter exists,
> every output traces back to dataset columns or parameters, and every input has
> a canonical `dataset_column`.

Output format:

> Return only valid JSON with sections for `symbols`, `inputs`, `parameters`,
> `derived_variables`, `metrics`, `outputs`, `equations`, and `steps`.

## 2. Auditor Prompt

Source: `prompts/auditor.yaml`

Purpose:

> You are a scientific model auditor and implementation planner. Your goal is to
> revise the extracted template and produce an implementation-ready plan for the
> code generator.

Inputs:

> Given: original paper text and extracted template.

Audit targets:

> Find missing variables, missing equations, missing parameters, missing outputs,
> missing dependencies in existing items, and a deterministic code-generation plan
> for the revised complete template.

Patch-only behavior:

> Identify ONLY items that are present in the paper but absent from the template.
> Do NOT repeat items that already appear in the template in the patch arrays.

Type requirement:

> Every missing symbol MUST include a `type` field set to exactly one of:
> `scalar`, `vector`, `matrix`, `array`, `string`, or `datetime`, inferred from
> the symbol's subscripts and definition.

Code-generation plan fields:

> The plan MUST cover existing template items and missing patch items. Required
> sections are `input_mapping`, `tariff_mapping`, `parameter_initialization`,
> `execution_order`, `pseudocode`, `result_dict_keys`, and `validation_checks`.

Implementation constraints:

> The pseudocode must be deterministic and directly translatable into pandas code.
> Do not introduce any variable, equation, parameter, tariff, or assumption that
> is not present in the paper or extracted template. Each pseudocode step must
> reference the equation id or variable name it implements.

Output format:

> Return only valid JSON: a patch object with missing elements and
> `code_generation_plan`. If nothing is missing, return empty patch arrays and
> still populate `code_generation_plan` from the extracted template.

## 3. Code-Generation Prompt

Source: `prompts/codegen.yaml`

Purpose:

> You are an expert Python programmer implementing an energy-community business
> model from a research paper. Write a single Python function that executes the
> model.

Function contract:

```python
def simulate(df, tariffs, parameters):
    ...
    return {...}
```

Arguments:

> `df` is a long-format pandas DataFrame with one row per member and timestep.
> `tariffs` is a pandas DataFrame loaded from `tariffs.csv`, containing a `time`
> column and hourly tariff columns. `parameters` is a dictionary of scalar
> overrides.

Return contract:

> Return a plain Python dictionary. Keys must use the paper's own variable names.
> Include every named intermediate result as well as all final outputs. Values
> must be JSON serializable.

Implementation priority:

> Implement strictly in this order: equations are authoritative; dependencies
> define execution order; the auditor plan is followed when consistent with
> equations and dependencies; `steps[]` is followed only when consistent; if
> pseudocode contradicts an equation, the equation wins.

Hard constraints:

> No file I/O, network calls, subprocesses, `eval`, `exec`, or global state.
> Never mutate the input dataframe directly. Tariff values must be read from
> exact tariff DataFrame columns and prices must never be hard-coded. Never invent
> numbers or model variables.

Mandatory robustness rules:

> Convert numeric inputs, tariff values, and numeric parameters with
> `pd.to_numeric(..., errors="coerce")`. Convert timestamps with
> `pd.to_datetime(..., errors="coerce")`. Remove invalid timestamps and sort by
> `Timestamp` before calculations. Guard dependencies before every equation.
> Use `safe_divide(a, b)` for all model divisions.

Required helper functions:

> Define nested helpers: `safe_numeric`, `safe_datetime`, `safe_divide`,
> `safe_get_parameter`, and `to_jsonable`.

Canonical-column rule:

> The only guaranteed input columns are `Timestamp`, `member_id`, `member_type`,
> `load`, and `generation`. Every other quantity is derived and must be computed
> before use.

Equation rules:

> Every equation in `equations[]` must have a corresponding implementation.
> Equation IDs must appear as comments above the implementing code line. Every
> derived variable or symbol with dependencies must be computed exactly once from
> those dependencies.

Data validation sequence:

> Validate required columns, copy the dataframe, parse timestamps, convert load
> and generation, drop invalid timestamps, validate member ids, sort by timestamp,
> detect duplicate `(member_id, Timestamp)` rows, and validate all required tariff
> columns before reading them.

Output instruction:

> Return only the Python function. No prose, no reasoning, no markdown fences.
> The response must start with `def simulate(df, tariffs, parameters):`.

## 4. Code-Auditor Prompt

Source: `prompts/code_auditor.yaml`

Purpose:

> You are a strict code auditor AND fixer for energy-community business model
> simulations.

Inputs:

> A business model template describing equations, parameters, and expected outputs,
> and a generated Python function named `simulate(df, tariffs, parameters)`.

Main job:

> Find every violation AND return a corrected version of the function.

Checks performed:

1. Missing equations: every equation id in `equations[]` must appear as a comment
   and be implemented.
2. Missing outputs: every name in `outputs[]` must be a key in the returned dict.
3. Unused parameters: every name in `parameters[]` must be read via
   `parameters.get("...", ...)`.
4. Unsatisfied dependencies: dependencies must be computed before dependent
   variables.
5. Forbidden input columns: non-canonical dataframe columns cannot be read before
   being computed.
6. Undefined variables: variables cannot be used before assignment.
7. Hardcoded or invented values: numeric constants must trace to an equation,
   parameter default, or tariff value.
8. Invented variables: returned or computed model variables must exist in the
   template symbol space or equation outputs.

Fixing rules:

> Produce a corrected version of the entire `simulate(df, tariffs, parameters)`
> function. Keep deterministic correct lines unchanged. Fix missing equations from
> template equations, add missing outputs, read unused parameters with
> `parameters.get`, compute forbidden or undefined columns from canonical inputs,
> trace hardcoded values to equations or parameters, and guard divisions.

Output format:

> Return only JSON with exactly two keys: `violations` and `fixed_code`.
> If there are no violations, set `violations` to an empty array and return the
> original function verbatim in `fixed_code`.

