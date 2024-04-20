async function getPhitterContinuousCode() {
    const resonse = await fetch("../../../../phitter_web/discrete/phitter_web_discrete.py");
    const discreteCode = await resonse.text();
    return discreteCode;
}

async function main() {
    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full",
    });
    await pyodide.loadPackage(["scipy"]);

    const discreteCode = await getPhitterContinuousCode();
    await pyodide.runPython(discreteCode);

    pyodide.runPython(`
        import time
        data = BINOMIAL(init_parameters_examples=True).sample(100000)

        ti = time.time()
        print("Init")
        phitter_discrete = PHITTER_DISCRETE(data=data)
        phitter_discrete.fit()
        tf = time.time()
        print(f"Execution time: {tf - ti}")

        sorted_distributions_sse = phitter_discrete.sorted_distributions_sse
        not_rejected_distributions = phitter_discrete.not_rejected_distributions

        for distribution, results in not_rejected_distributions.items():
            print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
    `)
}

main();

