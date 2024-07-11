#!/bin/bash

# Check if the input file parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Define the input file name from the parameter
input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found!"
    exit 1
fi

# Define the output file name
output_file="${input_file%.*}_speedup.${input_file##*.}"

# Read the execution time for 1 thread
base_time=$(awk -F, 'NR==1 {print $2}' "$input_file")

# Create the output file and write the header
echo "Threads,Time,Speedup" > "$output_file"

# Process each line of the input file
while IFS=, read -r threads time; do
  # Calculate the speedup
  speedup=$(echo "scale=2; $base_time / $time" | bc)
  efficiency=$(echo "scale=2; $speedup / $threads" | bc)
  
  # Write the results to the output file
  echo "$threads,$time,$speedup,$efficiency" >> "$output_file"
done < "$input_file"

echo "Output written to $output_file"
