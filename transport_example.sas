begin_version
3
end_version
begin_metric
0
end_metric
3
begin_variable
var0
-1
2
Atom at(truck-1, city-loc-1)
Atom at(truck-1, city-loc-2)
end_variable
begin_variable
var1
-1
2
Atom capacity(truck-1, capacity-0)
Atom capacity(truck-1, capacity-1)
end_variable
begin_variable
var2
-1
3
Atom at(package-1, city-loc-1)
Atom at(package-1, city-loc-2)
Atom in(package-1, truck-1)
end_variable
0
begin_state
0
1
1
end_state
begin_goal
1
2 0
end_goal
6
begin_operator
drive truck-1 city-loc-1 city-loc-2
0
1
0 0 0 1
50
end_operator
begin_operator
drive truck-1 city-loc-2 city-loc-1
0
1
0 0 1 0
50
end_operator
begin_operator
drop truck-1 city-loc-1 package-1 capacity-0 capacity-1
1
0 0
2
0 2 2 0
0 1 0 1
1
end_operator
begin_operator
drop truck-1 city-loc-2 package-1 capacity-0 capacity-1
1
0 1
2
0 2 2 1
0 1 0 1
1
end_operator
begin_operator
pick-up truck-1 city-loc-1 package-1 capacity-0 capacity-1
1
0 0
2
0 2 0 2
0 1 1 0
1
end_operator
begin_operator
pick-up truck-1 city-loc-2 package-1 capacity-0 capacity-1
1
0 1
2
0 2 1 2
0 1 1 0
1
end_operator
0