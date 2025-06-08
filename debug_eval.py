import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from tqdm import tqdm

with open("public_cases.json") as f:
    test_data = json.load(f)

def run_case(i, case):
    td = case["input"]["trip_duration_days"]
    miles = case["input"]["miles_traveled"]
    receipts = case["input"]["total_receipts_amount"]
    expected = case["expected_output"]

    try:
        result = subprocess.run(
            ["./run.sh", str(td), str(miles), str(receipts)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3
        )
        output = float(result.stdout.decode().strip())
        error = abs(expected - output)
        return (i, td, miles, receipts, expected, output, error)
    except:
        return (i, td, miles, receipts, expected, None, None)

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(run_case, i, case) for i, case in enumerate(test_data)]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())

# Sort by error
results = [r for r in results if r[6] is not None]
results.sort(key=lambda x: -x[6])

# Print top 10 high-error cases
print("\n🔍 Top 10 High-Error Cases:")
for i, td, miles, rec, exp, out, err in results[:10]:
    print(f"Case {i+1}: {td} days, {miles} miles, ${rec:.2f} receipts")
    print(f"  → Expected: ${exp:.2f}, Got: ${out:.2f}, Error: ${err:.2f}")
