# Description: Run all experiments
# Usage: bash scripts/all.sh

echo "Running all experiments"
echo "Running imp_interval_impact"
bash scripts/imp_interval_impact.sh 50 350 imp_interval_impact


echo "Running imp_m_impact"
bash scripts/imp_m_impact.sh 1 40 imp_m_impact

echo "Running imp_period_length_impact"
bash scripts/imp_period_length_impact.sh 100 1000 imp_period_length_impact

echo "Running imp_year_impact"
bash scripts/imp_year_impact.sh 20 200 imp_year_impact

echo "Running imp_parallel"
bash scripts/imp_parallel.sh 1 30 imp_parallel