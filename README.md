# ev_cost-time-optimisation
to generate grid network:
netgenerate --grid --grid-number 5 --length 100 --output my_grid.net.xml
to generate trips :
use random trips file 
python $SUMO_HOME/tools/randomTrips.py -n my_grid.net.xml -o trips.trips.xml --period 5
to run on sumo
 sumo-gui -n my_grid.net.xml -r trips.trips.xml -e 100

generated new_charging.add.xml using generator.py 
