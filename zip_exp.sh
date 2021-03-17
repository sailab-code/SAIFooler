
for t in $@; do
  zip -r $t.zip ./logs_09_march/$t/pgd_attack/*
done

