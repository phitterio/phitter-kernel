async function getPhitterContinuousCode() {
    const resonse = await fetch("../../../../phitter_web/continuous/phitter_web_continuous.py");
    const continuousCode = await resonse.text();
    return continuousCode;
}

async function main() {
    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full",
    });
    await pyodide.loadPackage(["scipy"]);

    const continuousCode = await getPhitterContinuousCode();
    await pyodide.runPython(continuousCode);

    pyodide.runPython(`
        import time
        data = BETA(init_parameters_examples=True).sample(1000000)

        ti = time.time()
        print("Init")
        phitter_continuous = PHITTER_CONTINUOUS(data=data, subsample_estimation_size=10000)
        phitter_continuous.fit()
        tf = time.time()
        print(f"Execution time: {tf - ti}")

        sorted_distributions_sse = phitter_continuous.sorted_distributions_sse
        not_rejected_distributions = phitter_continuous.not_rejected_distributions

        for distribution, results in not_rejected_distributions.items():
            print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
    `)
}

main();

