# Set the data file separator
set datafile separator ","
# Get the filenames from the command-line arguments
file_sweep = ARG1
file_heap = ARG2

# Set the terminal type and output file if needed
set terminal pngcairo size 800,500 enhanced font 'Arial,16'
set output 'scripts/Figures/knn_exclusion_var.png'


# Set labels for the axes
set xlabel 'Exclusion size'
set ylabel 'Time (ms)'

# Set the key (legend) outside the plot area to avoid overlapping
set key inside top left


# Customize line styles for better visualization
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # Blue 
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # Red 


# Plot the data from both files
plot file_sweep using 1:2 with linespoints linestyle 1 title 'Linear sweeps', \
     file_heap using 1:2 with linespoints linestyle 2 title 'Heap-based'
