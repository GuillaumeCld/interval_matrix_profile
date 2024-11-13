print ARGC

set datafile separator ","
file = ARG1



# Set the terminal type and output file if needed
set terminal pdfcairo size 10in,5in enhanced font 'Latin Modern Roman,26'
set output 'scripts/Figures/imp_parallel.pdf'


# Set the title and labels
set xlabel "Number of Threads" font 'Latin Modern Roman,28'
set ylabel 'Time (min)' font 'Latin Modern Roman,28' 
set y2label "Speedup"

set grid

set key center top font 'Latin Modern Roman,26' spacing 1 width 1 box

set xtics 0,6,30 font 'Latin Modern Roman,24' nomirror 
set ytics 0,2,10 font 'Latin Modern Roman,24' nomirror
set y2tics 0,6,30 font 'Latin Modern Roman,24' nomirror 

set xrange [1:30] writeback
set y2range [1:31] writeback
set yrange [0:10] writeback

# Customize line styles for better visualization
set style line 1 lc rgb '#006400' lt 1 lw 4 pt 9 ps 1.75   # Green with triangle points
set style line 2 lc rgb '#FF8C00' lt 1 lw 4 pt 13 ps 1.75   # Red with circle points

# Plot the data
plot file using 1:($2/60) with linespoints linestyle 1 title "Execution Time", \
     "" using 1:3 with linespoints linestyle 2 title "Speedup" axes x1y2
