from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np

def ret():
    session = fastf1.get_session(2019, 'Monza', 'Race')
    session.load()

    laps = session.laps
    car_data = session.car_data

    drivers = {}

    for i in session.drivers:
        driver = session.get_driver(i).loc[["DriverId","Abbreviation","GridPosition"]]

        laps_info = laps[laps.Driver == driver.Abbreviation]
        car_info = car_data[i]
        #car_info = car_info.drop([x for x in range(car_info.shape[0]) if x % 5 == 0])

        last_lap = session.session_start_time
        last_position = driver.GridPosition
        n_lap = 0
        stats_laps = []
        #falta detetar se se espetou
        # verificar qualidade de dados para filtrar
        for _,lap in laps_info.iterrows():
            stats_lap = {}

            car_info_lap = car_info[np.logical_and(last_lap < car_info.Time, car_info.Time < lap.Time)] 
            mean_speed = int(car_info_lap.Speed.mean())

            stats_lap["Speed"] = mean_speed
            stats_lap["LapTime"] = lap.Time - last_lap
            stats_lap["CurrentPosition"] = lap.Position 
            stats_lap["DeltaPosition"] = lap.Position - last_position   

            stats_laps.append(stats_lap)
            n_lap = n_lap + 1

            last_lap = lap.Time
            last_position = lap.Position

        driver["Lap"] = stats_laps
        drivers[i] = driver

    print(drivers["16"].Lap)
    print(len(drivers["16"].Lap))
    return drivers