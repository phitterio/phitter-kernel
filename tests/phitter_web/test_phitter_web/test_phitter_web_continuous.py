import sys

sys.path.append("../../../phitter_web/continuous/")
from phitter_web_continuous import PHITTER_CONTINUOUS


def get_data(path: str) -> list[float | int]:
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()
    return data


path = "../../../datasets_test/continuous/data_1000/data_alpha.txt"
# path = "../datasets_test/discrete/book2.txt"
# path = "../datasets_test/continuous/data_1000/data_beta.txt"

data = get_data(path)

phitter_continuous = PHITTER_CONTINUOUS(data)
phitter_continuous.fit(n_workers=1)

for distribution, results in phitter_continuous.sorted_distributions_sse.items():
    print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
