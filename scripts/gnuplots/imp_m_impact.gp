print ARGC

# Set the file input and separator
set datafile separator ","
file_bf = ARG1
file_stomp = ARG2
file_block = ARG3

# Set the terminal type, size, and output file for the plot
set terminal pdfcairo size 10in,4.5in enhanced font 'Latin Modern Roman,26'
set output 'scripts/Figures/imp_m_impact.pdf'

# Set labels for the axes and add grid lines
set xlabel '{/:Italic m}'  font 'Latin Modern Roman,28'
set ylabel 'Time (s)' font 'Latin Modern Roman,28'
set grid

set key left top font 'Latin Modern Roman,26' spacing 1 width 2.5 box

set xtics font 'Latin Modern Roman,24' nomirror 
set ytics 1 font 'Latin Modern Roman,24' nomirror

# Custom line styles 
set style line 1 lc rgb '#0060ad' lt 1 lw 4 pt 5 ps 1.75   # Blue with sqaure points
set style line 2 lc rgb '#dd181f' lt 1 lw 4 pt 7 ps 1.75   # Red with circle points
set style line 3 lc rgb '#006400' lt 1 lw 4 pt 9 ps 1.75   # Green with triangle points


# Plot the data from all files
plot file_stomp using 1:($2/1000) with linespoints linestyle 2 title 'Modified AAMP', \
     file_bf using 1:($2/1000) with linespoints linestyle 1 title 'Brute Force', \
     file_block using 1:($2/1000) with linespoints linestyle 3 title 'BIMP'
