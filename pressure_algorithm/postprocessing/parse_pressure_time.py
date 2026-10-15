import re
import csv


def parse_pressure_file(filename):
    """
    Parses a pressure report file and returns two lists:
    - times: list of floating point step times
    - pressures: list of corresponding first pressure values (column P)
    """
    times = []
    pressures = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Look for line containing "Step Time ="
        match = re.search(r'Step Time\s*=\s*([\d\.Ee\+\-]+)', line)

        if match:
            time_val = float(match.group(1))

            # Skip lines until we hit the dashed separator line
            j = i + 1
            while j < n and not re.match(r'^-+', lines[j].strip()):
                j += 1

            # After the separator, the next non-empty line is the first data row
            j += 1
            while j < n and lines[j].strip() == '':
                j += 1

            if j < n:
                data_line = lines[j].strip()
                parts = data_line.split()

                if parts:
                    # The last column is the pressure value
                    p_val = float(parts[-1])
                    times.append(time_val)
                    pressures.append(p_val)

            # Continue searching after this block
            i = j
        else:
            i += 1

    return times, pressures


def save_pressure_csv(times, pressures, output_filename):
    """
    Saves times and pressures to a CSV file with columns:
    t, p
    """
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'p'])

        for t, p in zip(times, pressures):
            writer.writerow([t, p])


# Example usage:
if __name__ == "__main__":
    input_file = r"data/forming/Free_bulg_test/pressure_algo.rpt"
    output_file = r"data/forming/Free_bulg_test/pressure_algo_tp.csv"

    t, p = parse_pressure_file(input_file)

    save_pressure_csv(t, p, output_file)

    print("Times:", t[:5], "...")
    print("Pressures:", p[:5], "...")
    print("Saved to:", output_file)