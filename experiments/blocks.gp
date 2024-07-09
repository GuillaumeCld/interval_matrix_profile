# Set the terminal type to png and specify the output file
set terminal pngcairo enhanced size 800,600
set output 'computation_times.png'

# Set titles and labels
set title "Computation Time vs Block Dimensions"
set xlabel "Block Dimension (Width x Height)"
set ylabel "Computation Time (seconds)"

# Set grid
set grid

# Set xtics to show both block width and height
set xtics rotate by -45
set xtics font ",8"

# Read the data file
set datafile separator ","

# Set the format for xtics labels
set xtics format '(%0.f, %0.f)'

# Plot the data
plot "computation_times.txt" using (stringcolumn(1) . "x" . stringcolumn(2)):(column(5)) with linespoints title "Computation Time"

# Optional: Save the plot as a PDF
set terminal pdf enhanced
set output 'computation_times.pdf'
replot
