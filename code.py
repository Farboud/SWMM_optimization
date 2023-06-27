import os
import subprocess
import datetime
import numpy as np
import datetime
import math
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

FNULL = open(os.devnull, 'w')    #use this if you want to suppress output to stdout from the subprocess

def flooding_summary(filename):
    # Reads flooding in nodes from SWMM report file
    # Open the input file
    with open(filename, "r") as file:
        data = file.readlines()

    # Find the start and end indices
    start_index = 0
    end_index = len(data) - 1

    # Find the index of "Node Flooding Summary" followed by "***********************"
    for i, line in enumerate(data):
        if line.strip() == "Node Flooding Summary" and data[i + 1].strip() == "*********************":
            start_index = i
            break

    # Find the index of the next line with "***********************" after the start line
    for i, line in enumerate(data[start_index:]):
        if line.strip() == "***********************":
            end_index = start_index + i
            break

    # Read the contents between the start and end indices
    contents = data[start_index:end_index + 1]  # Include the end line

    # Check if "No nodes were flooded." is found
    if "  No nodes were flooded.\n" in contents:
        control_array = ['48', '49', '143', '145', '147', '148', '157', '158', '47a', '47b', '159']
        node_names = control_array
        flood_volume = [0] * len(node_names)
        flood_time = [0] * len(node_names)
    else:
        # Skip 10 lines
        table_start_index = start_index + 10

        # Tabulate the data
        table_data = []
        for line in data[table_start_index:end_index]:
            line_data = line.strip().split()
            table_data.append(line_data)

        # Remove the last two rows from the tabulated data
        table_data = table_data[:-2]

        # Process and store the relevant information
        node_names = [row[0] for row in table_data]
        flood_volume = [float(row[5]) for row in table_data]
        flood_time = [float(row[1]) for row in table_data]

        # Check missing values from the control array
        control_array = ['48', '49', '143', '145', '147', '148', '157', '158', '47a', '47b', '159']
        for value in control_array:
            if value not in node_names:
                index = control_array.index(value)
                node_names.insert(index, value)
                flood_volume.insert(index, 0)
                flood_time.insert(index, 0)

    # Return the node names and flood volume and flood time
    return node_names, flood_volume, flood_time


def inflow_summary(filename, node_names):
    #Reads inflow summary from SWMM report file and return the inflow volume for flooded nodes.
    # Open the input file
    with open(filename, "r") as file:
        data = file.readlines()

    # Find the index of the "Node Inflow Summary" section
    start_index = 0
    for i, line in enumerate(data):
        if line.strip() == "Node Inflow Summary":
            start_index = i + 8  # Skip the next eight lines
            break

    # Tabulate the data until a line with "*********************" is found
    table_data = []
    for line in data[start_index:]:
        if "*********************" in line:
            break
        table_data.append(line.strip().split())

    # Remove the last two rows from the tabulated data
    table_data = table_data[:-2]

    # Filter the tabulated data based on the desired node names and extract the 8th column
    inflow_volume = [float(row[7]) for row in table_data if row[0] in node_names]

    # Return the filtered data
    return inflow_volume



def calculate_flood_p0(flood_volume, inflow_data):
    #Finds ratio volume flooded based on volume of inflow and volume of flooding
    # Ensure that both lists have the same length
    if len(flood_volume) != len(inflow_data):
        print("Error: The lists have different lengths.")
        return None
    
    flood_p0 = []
    for i in range(len(flood_volume)):
        try:
            fv = float(flood_volume[i])
            idata = float(inflow_data[i])
            if idata != 0:
                division = fv / idata
                flood_p0.append(division)
            else:
                flood_p0.append(None)  # Handle division by zero case
        except ValueError:
            flood_p0.append(None)  # Handle invalid float conversion case

    return flood_p0

def input_creator(inp_name, intensity_24hr, time, scs_type_3_intensity, return_period):
    #adds rainfall data to SWMM input file based on intensity
    intensity_series = [i * intensity_24hr[return_period] * 25.4 / 10 for i in scs_type_3_intensity]  # Conversion to mm/hr

    # Create a new input file
    with open(inp_name + '_1.inp', "w") as f:
        # Write the first part of the input template
        with open("input_part_1_lid.inp", "r") as f1:
            fc = f1.read()
            f.write(fc)

        # Write the new dataset
        f.write("; Method = SCS type III, Intensity=" + str(intensity_24hr[return_period] * 25.4) + " mm/hr\n")
        for i in range(len(intensity_series)):
            f.write("SCS " + time[i] + " " + f"{intensity_series[i]:.3f}" + "\n")

        # Write the second part of the input template
        with open("input_part_2_lid.inp", "r") as f1:
            fc = f1.read()
            f.write(fc)
            
def generate_timestamp(sim_time):
    timestep = 0.1  # unit = hour (6 minutes = 0.1 hour)
    timestep= int(60*timestep) #conversion to min
    time = []

    for h in range(sim_time):
        for m in range(0, 60, timestep):
            x = datetime.datetime(2018, 9, 1, h, m, 0)
            time.append(x.strftime("%H:%M"))

    time.append(f"{sim_time:02d}:00")#last number in the series= total sim time
    return time

def logarithm_base10(arr):
    result = []
    for row in arr:
        temp_row = []
        for num in row:
            if num == 0 or num == 1:
                temp_row.append(0)
            else:
                temp_row.append(math.log10(num))
        result.append(temp_row)
    return result

def antifragility(p0):
    logp0 = logarithm_base10(p0)
    p0logp0 = np.multiply(p0, logp0)
    p1 = np.ones_like(p0) - p0
    logp1 = logarithm_base10(p1)
    p1logp1 = np.multiply(p1, logp1)
    k = 1
    h = -1 * (p0logp0 + p1logp1) * k  # information entropy based on log10
    h_bar = np.mean(h, axis=0)
    d = (p0 - 0.5) ** 2 + (p1 - 0.5) ** 2
    d_bar = np.mean(d, axis=0)
    c_bar = h_bar * d_bar
    delta_c = c_bar - c_bar[0]
    #for optimization purpases delta c suffices
    
    return delta_c

def update_file_lid_section(filename, subcatchments, area_values):
    #writes the LID area to the input file
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the start and end indices of the section
    start_index = None
    end_index = None

    for i, line in enumerate(lines):
        if line.strip() == "[LID_USAGE]":
            start_index = i + 1
        elif start_index is not None and line.strip() == "":
            end_index = i
            break

    if start_index is None or end_index is None:
        raise ValueError("Section not found in the file.")

    # Update all "Area" values to 0
    for i in range(start_index, end_index):
        line = lines[i]
        values = line.strip().split()
        values[3] = "0"  # Update the "Area" value to 0
        lines[i] = " ".join(values) + "\n"

    # Update the "Area" values based on the provided lists
    for subcatchment, area_value in zip(subcatchments, area_values):
        for i in range(start_index, end_index):
            line = lines[i]
            values = line.strip().split()
            if values[0] == subcatchment:
                values[3] = str(area_value)  # Update the "Area" value
                lines[i] = " ".join(values) + "\n"
                break

    # Write the updated lines back to the file
    with open(filename, 'w') as file:
        file.writelines(lines)

def swmm(lid):
    #calculates delta_c for antifragility and reliability from SWMM model.
    #only one of the above is used as the putput for the purpose of optimization based on that one parameter.
    # Start
    inp_name = "Calibration_daily"
    output_name = "f"
    sim_time = 6  # hours

    # Precipitation intensity estimates with 90% confidence intervals for Columbia (in)
    return_periods = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

    # Data source: https://hdsc.nws.noaa.gov/pfds/pfds_map_cont.html?bkmrk=sc
    intensity_6hr = [2.20, 2.62, 3.21, 3.79, 4.57, 5.27, 6.01, 6.81, 7.95, 8.98]

    junctions = ['48', '49', '143', '145', '147', '148', '157', '158', '47a', '47b', '159']
    p0 = np.empty((len(return_periods), len(junctions)))
    reliability_array = np.empty((len(return_periods), len(junctions)))

    # Create 6hr rainfall intensities
    scs_type_3_intensity = [0, 0.045, 0.046, 0.045, 0.045, 0.045, 0.046, 0.046, 0.048, 0.051, 0.054, 0.055, 0.058, 0.063,
                            0.065, 0.068, 0.072, 0.076, 0.082, 0.088, 0.093, 0.1, 0.108, 0.122, 0.136, 0.15, 0.163, 0.336,
                            0.674, 1.081, 1.94, 1.346, 0.269, 0.235, 0.203, 0.173, 0.139, 0.119, 0.113, 0.104, 0.097, 0.09,
                            0.085, 0.079, 0.077, 0.073, 0.068, 0.065, 0.063, 0.059, 0.057, 0.053, 0.053, 0.051, 0.049,
                            0.048, 0.048, 0.047, 0.047, 0.045, 0.044]  # unit = 1/hr

    # Create timestamps
    time = generate_timestamp(sim_time)
    subcatchments = ["ws-43", "ws-45", "ws-46", "ws-47", "ws-48", "ws-49", "ws-50", "ws-52", "ws-142", "ws-143",
                     "ws-144", "ws-145", "ws-146", "ws-148", "ws-149", "ws-150", "ws-155", "ws-156", "ws-157", "ws-159"]

    update_file_lid_section("input_part_1_lid.inp", subcatchments, lid)

    # Iterate return periods
    for period in range(len(return_periods)):
        # Create SWMM input file with the appropriate rainfall data
        input_creator(inp_name, intensity_6hr, time, scs_type_3_intensity, period)

        # Run the model with the given input file
        args = "swmm5.exe " + '"' + inp_name + '_1.inp" ' + output_name + ".rpt " + output_name + ".out"
        subprocess.call(args, stdout=FNULL, stderr=FNULL, shell=False)

        node_names, flood_volume, flood_time = flooding_summary(output_name + ".rpt")
        inflow_volume = inflow_summary("f.rpt", junctions)
        flood_p0 = calculate_flood_p0(flood_volume, inflow_volume)
        p0[period] = np.array(flood_p0)

        # Reliability
        reliability = []

        for x, y, z in zip(flood_volume, inflow_volume, flood_time):
            if y == 0 or sim_time == 0:
                reliability.append(1)  # Avoid division by zero
            else:
                reliability.append((1 - (x / y)) * (1 - (z / sim_time)))

        reliability_array[period] = np.array(reliability)
    
    reliability = np.mean(reliability_array, axis=1)
    # Antifragility
    delta_c = antifragility(p0)

    #Change the output of the function between delta_c or reliability as needed.
    return delta_c[-2]

def constraint(x):
    # Define the constraint function here
    #Max impervious area in subcatchment (total area * impervious percentage) is used
    max_values = [4117, 76162, 129663, 3899, 2186, 10066, 6205, 10164, 205, 1172, 10009, 630, 781, 410145, 8, 183471, 505, 2455, 6155, 300607]   # Maximum values for each item
    constant_sum = np.sum(max_values)*0.05 #assuming max 5% of impervious area to be used
    return np.sum(x) - constant_sum  # constant_sum is the desired sum



start_time = time.time()
current_time_formatted = time.strftime("%H:%M", time.localtime())
print("Current time is:", current_time_formatted)


# Define the bounds for each variable in the array
min_values = 20*[0]  # Minimum values for each item
#Max area for each subcatchment in m2
max_area = [7500, 152000, 172600, 6600, 2700, 11100, 6300, 16700, 1500, 6900, 18900, 1000, 2600, 814300, 300, 445300, 1600, 5200, 15700, 743700]   # Maximum values for each item
#Max impervious area in subcatchment (total area * impervious percentage) is used
max_values = [4117, 76162, 129663, 3899, 2186, 10066, 6205, 10164, 205, 1172, 10009, 630, 781, 410145, 8, 183471, 505, 2455, 6155, 300607]   # Maximum values for each item
# Define the number of variables (array size) and constant sum
num_variables = len(min_values)
#constant_sum = np.sum(max_values)*0.05 #assuming max 5% of impervious area to be used (for the case of PP)
constant_sum = 15879.2 #equivalent area for the same amount of money as in the PP case
   
# Set up the linear constraint for the constant sum
linear_constraint = LinearConstraint(np.ones((1, num_variables)), [constant_sum], [constant_sum])
    
# Define the bounds for each variable
bounds = []
for i in range(num_variables):
    bounds.append((min_values[i], max_values[i]))
    
# Define the initial guess for the variables
x0 = np.random.uniform(min_values, max_values, num_variables)
x0 = [1.05E+01, 1.85E+01, 1.35E+01, 1.12E+01, 2.63E+00, 5.15E+00, 8.90E+00, 1.04E+01, 6.07E-01, 3.07E+00, 3.14E+01, 1.28E+00, 2.57E+00, 1.57E+04, 2.85E-02, 9.73E+00, 1.78E+00, 2.42E+00, 1.66E+01, 1.36E+01]
# Perform the optimization
result = minimize(swmm, x0, method='SLSQP', constraints=[linear_constraint], bounds=bounds)    


current_time = time.time() - start_time
print("Runtime:", current_time)

# Print the optimized variables and the corresponding minimum value of F
print("Optimized variables:", result.x)
##############################
##############################
##############################
