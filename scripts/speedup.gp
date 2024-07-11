# Define variables for input and output files
input_file = ARG1
output_file = ARG2

# Set the terminal and output file
set terminal pngcairo size 800,500 enhanced font 'Arial,16'
set output "para.png"

# Set the data file separator
set datafile separator ","

# Set the title and labels
# set title "Brute Force Execution Time and Speedup vs. Number of Threads"
set xlabel "Number of Threads"
set ylabel "Time (s)"
set y2label "Speedup"

# Enable grid and logscale for the primary y-axis
set grid
# set logscale y

# Set the key (legend) outside the plot area to avoid overlapping
set key inside center right


# Set ticks axis
set xtics nomirror
set ytics nomirror
# Enable a secondary y-axis on the right side for speedup
set y2tics

# Customize line styles for better visualization
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # Blue for execution time
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # Red for speedup

# Plot the data
plot "para_stomp_speedup.txt" using 1:2 with linespoints linestyle 1 title "Execution Time", \
     "" using 1:3 with linespoints linestyle 2 title "Speedup" axes x1y2
