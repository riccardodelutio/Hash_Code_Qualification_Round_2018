# -*- coding: utf-8 -*- 
"""
HashCodeQualification

Team HASH-ME

team members :

Aymeric Dieuleveut
Evann Courdier
Riccardo de Lutio

Spotted a tiny mistake the day after submission (see find_best_ride function). After fixing our submission was ranked 130 / 3500
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True)

file_a = open('a_example.in', 'r').read()
file_b = open('b_should_be_easy.in', 'r').read()
file_c = open('c_no_hurry.in', 'r').read()
file_d = open('d_metropolis.in', 'r').read()
file_e = open('e_high_bonus.in', 'r').read()

"""### The following function is to parse an input file"""

def parse_input(file):
    result = [line.split(' ') for line in file.split('\n')]
    header = list(map(int, result[0]))
    result = result[1:-1][:]
    result = np.array(result, dtype=int)
    distance = np.abs(result[:,2] - result[:,0]) + np.abs(result[:,3]-result[:,1])
    latest_start = result[:,-1] - distance - np.ones(result.shape[0])
    index_rides = np.arange(header[3], dtype=int)
    result = np.hstack((result,(distance).reshape((result.shape[0],1)),(latest_start).reshape((result.shape[0],1)),(index_rides).reshape((result.shape[0],1))))
    return result,header

"""### The following function is to write a solution to a file and download it"""

def write_download(solution, name='output.txt'):
    file = open(name, 'w')
    for l in solution:
        file.write(str(len(l)) + ' ' + ' '.join(map(str, l)) + '\n')
    file.close()

"""# ======== CODE BELOW ========"""

"""
rides :

Rows index corresponds to ride number

Columns definitions :

0.   row_start
1.   col_start
2.   row_end
3.   col_end
4.   start_time
5.   latest_end
6.   distance_ride
7.   latest_start
8.   index_rides
"""

rides, header = parse_input(file_a) # Choose here file you want to test

"""
Header definition :

0.   num_rows
1.   num_cols
2.   num_vehicules
3.   num_rides
4.   bonus
5.   t_max
"""

bonus = header[4]
car_numbers = header[2]
num_rides = header[3]
t_max = header[5]

"""
fleet_matrix :

Rows index corresponds to car number

Columns definitions :

0.   row_pos
1.   col_pos
2.   time_free : first time where the car will not be assigned to any ride
"""

fleet_matrix = np.zeros((car_numbers, 3))
fleet_matrix[:,-1] = -1

"""
fleet_rides definition :

Rows index corresponds to car number

Each row contains list of rides assigned to specific car
"""

fleet_rides = [[] for i in range(car_numbers)]

"""
rides_available definition :

Boolean vector indicating if a ride is available or not

Index corresponds to ride number
"""

rides_available = np.ones(num_rides, dtype=bool)
rides = rides[:num_rides]


"""
find_best_ride :

finds the best ride for a given car
"""
def find_best_ride(car, rides, rides_available):
    row_car, col_car, current_time = car
    lowest_cost = -1
    best_ride = -1

    for ride in rides[rides_available]: # For all rides that are available
        start_pos = ride[:2]
        latest_start = ride[7]
        ride_index = ride[8]
        ride_start_time = ride[4]

        dist = abs(row_car - start_pos[0]) + abs(col_car - start_pos[1]) # Compute distance between car position and start position

        if dist + current_time <= latest_start:
            wait_time = ride_start_time - current_time - dist - 1 # Compute the time the car has to wait before the ride starts

            if wait_time >= 0:
                wait_time -= bonus # If the car has to wait we want to decrease the cost of waiting by the bonus because the ride will start on time
            else: # This was the mistake we didn't assign the value 0 to the wait time if the car arrived after start time
                wait_time = 0

            cost = dist + wait_time # The cost of a ride corresponds to the time to get to the ride start (which is same as distance to the ride start) plus the waiting time

            if cost < lowest_cost or lowest_cost == -1: # Save the ride corresponding to the lowest cost
                lowest_cost = cost
                best_ride = ride_index
    return int(best_ride)

"""
find_best_car :

finds the best car this is simply the argmin of the time_free vector
"""

def find_best_car(fleet_matrix, time_max):
    index_best_car = np.argmin(fleet_matrix[:,-1])
    best_car = fleet_matrix[index_best_car]
    if best_car[-1] < time_max:
        return best_car, index_best_car
    else :
        print("Nothing to do") # This should never happen! End condition is already taken care in the run loop
        return -1, -1

"""
update_car :

Once we have chosen the best car and it's best ride we have to update the fleet_matrix
Also returns has_bonus boolean for the scoring function (which doesn't work perfectly)
"""

def update_car(fleet_matrix,best_ride,index_best_car):
    distance = np.abs(fleet_matrix[index_best_car,0]-best_ride[0])+np.abs(fleet_matrix[index_best_car,1]-best_ride[1])
    fleet_matrix[index_best_car,0:2] = best_ride[2:4] # updates position of car to end position of the assigned ride
    fleet_matrix[index_best_car,2] += distance # the first time where it will be available again has to be increased by the distance to the start
    has_bonus = False
    wait_time = best_ride[4] - fleet_matrix[index_best_car,2] - 1
    if (wait_time) >= 0: # if car has to wait before the start
        has_bonus = True
        fleet_matrix[index_best_car,2] += wait_time # if it does it inscreases by the wait_time
    fleet_matrix[index_best_car,2] += best_ride[6] # Also increases by the length of the ride
    return has_bonus

"""
update :

This runs one update, we want to find the earliest available car and assign it to the best ride possible
"""
def update(fleet_matrix,fleet_rides,rides,rides_available,t_max,score):
    best_car, index_best_car = find_best_car(fleet_matrix, t_max)
    best_ride_index = find_best_ride(best_car, rides, rides_available)
    has_bonus = update_car(fleet_matrix,rides[best_ride_index,:], index_best_car)
    rides_available[best_ride_index] = False # Ride is assigned so not available anymore
    if best_ride_index != -1:
        fleet_rides[index_best_car].append(best_ride_index) # We want to save the assigned rides
        score += rides[best_ride_index, 6]
        score += has_bonus
    else:
        fleet_matrix[index_best_car,2] = t_max # If the car hasn't got a best ride, we want to set it's time_free at t_max so that we won't test it again (avoid infinite loop)
    return score

# Main run loop simulation

min_time = 0
score = 0
it=0
while min_time < t_max: # Stopping criterion if all the time_free for every vehicule are set to t_max then there is nothing to do anymore
    score = update(fleet_matrix,fleet_rides,rides,rides_available,t_max,score)
    it+=1
    if it%100==0:
        print(it,score) # print every 100 the iteration number and score
    min_time = np.min(fleet_matrix[:,2])


print(score)  # Score function doesn't seem to work perfectly

write_download(fleet_rides)
