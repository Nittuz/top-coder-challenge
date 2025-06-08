import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from tqdm import tqdm

# Load test cases
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # capture stderr for debugging
            timeout=3
        )
        output_str = result.stdout.decode().strip()

        try:
            output = float(output_str)
            error = abs(expected - output)
            return (i, expected, output, error)
        except Exception as e:
            print(f"❌ Case {i+1} failed to parse output:")
            print(f"STDOUT: {output_str}")
            print(f"STDERR: {result.stderr.decode()}")
            return (i, expected, None, None)

    except subprocess.TimeoutExpired:
        print(f"❌ Case {i+1} timed out.")
        return (i, expected, None, None)

# Run all test cases in parallel
print("🚀 Starting parallel evaluation...")

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(run_case, i, case) for i, case in enumerate(test_data)]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())

# Filter and report
valid = [r for r in results if r[2] is not None]
errors = [r[3] for r in valid]

print(f"\n✅ Evaluation Complete!")
print(f"Total cases: {len(test_data)}")
print(f"Exact matches (±$0.01): {sum(e < 0.01 for e in errors)} ({100 * sum(e < 0.01 for e in errors)/len(test_data):.1f}%)")
print(f"Close matches (±$1.00): {sum(e < 1.00 for e in errors)} ({100 * sum(e < 1.00 for e in errors)/len(test_data):.1f}%)")

if errors:
    print(f"Average error: ${mean(errors):.2f}")
    print(f"Max error: ${max(errors):.2f}")
else:
    print("❌ No valid outputs — check run.sh or model loading.")
