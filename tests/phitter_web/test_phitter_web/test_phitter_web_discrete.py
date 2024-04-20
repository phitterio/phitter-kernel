import sys

sys.path.append("../../../phitter_web/discrete/")
from phitter_web_discrete import PHITTER_DISCRETE


def get_data(path: str) -> list[int]:
    sample_distribution_file = open(path, "r")
    data = [int(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()
    return data


path = "../../../datasets_test/discrete/sample_binomial.txt"
data = get_data(path)

phitter_discrete = PHITTER_DISCRETE(data)
phitter_discrete.fit()

for distribution, results in phitter_discrete.not_rejected_distributions.items():
    print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
