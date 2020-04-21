
for ds in telco iris turtles wine auto housing; do
    echo "$ds"
    for f in sql mllib; do
        if test -f "spark_pkg/$ds/$f.py"; then
            echo "    $ds/$f.py"
            python spark_pkg/$ds/$f.py > spark_pkg/$ds/logs/$f.log 2>/dev/null
        fi
    done
done