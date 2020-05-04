
for ds in telco iris turtles wine auto housing; do
    echo "$ds"
    # spark
    for f in sql mllib; do
       if test -f "spark_pkg/$ds/$f.py"; then
            echo "    spark: $ds/$f.py"
            python spark_pkg/$ds/$f.py > spark_pkg/$ds/logs/$f.log 2>/dev/null
        fi
    done
    # scikit-learn
    echo "    scikit-learn"
    python scikit_learn_pkg/$ds.py > scikit_learn_pkg/logs/$ds.log 2>/dev/null
    
done
