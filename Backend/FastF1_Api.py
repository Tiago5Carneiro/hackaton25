import fastf1
import pandas as pd
import numpy as np
import mysql.connector 
import datetime
import time

def get_year_events(year):
    events = []
    
    start = datetime.datetime(year,1,1)

    event_schedule = fastf1.get_events_remaining(start)

    event_name = event_schedule['EventName'].tolist()
    event_date = pd.to_datetime(event_schedule['EventDate'])
    
    current_date = datetime.datetime.now()
    current_year = current_date.year
    for event,date in zip(event_name,event_date):
        if year == current_year:
            if not current_date < date: 
                events.append(fastf1.get_event(year,event))
        else : 
            events.append(fastf1.get_event(year,event))

    return events

def get_races(events):
    print(events)
    races = []
    current_date = datetime.datetime.now()
    for event in events:
        if event.EventDate < current_date:
            races.append(event.get_race())
    print(races) 
    return races

def get_stats_race(session):
    print(session)
    session.load(messages = False)
    
    laps = session.laps
    car_data = session.car_data
    
    event_info = {}
    drivers = {}
    
    event_info["StartDate"] = session.session_info["StartDate"]
    event_info["EndDate"] = session.session_info["EndDate"]
    event_info["EventName"] = session.event.OfficialEventName
    event_info["Country"] = session.event.Country
    event_info["Location"] = session.event.Location
    event_info["Winner"] = session.results.iloc[0]
    event_info["FastLap"] = session.laps.pick_fastest()


    for i in session.drivers:
        results = session.results

        driver = session.get_driver(i).loc[["DriverId","Abbreviation","GridPosition","HeadshotUrl","FullName","TeamName","ClassifiedPosition"]]
        driver["Points"] = int(results[results.DriverNumber == i].Points)
    
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
            stats_lap["CurrentPosition"] = lap.Position
            stats_lap["DeltaPosition"] = lap.Position - last_position
    
            stats_laps.append(stats_lap)
            n_lap = n_lap + 1
    
            last_lap = lap.Time
            last_position = lap.Position
    
        driver["Lap"] = stats_laps
        drivers[i] = driver
    
    event_info["Drivers"] = drivers

    return event_info

if __name__ == "__main__":
    current_year = datetime.datetime.now().year

    player_stats = {}
    races = []

    flag = True

    for year in range(current_year,2017,-1):
        events = get_year_events(year)
        races = get_races(events)

        player_points = {}

        for race in races:
            stats = get_stats_race(race)

            drivers = stats["Drivers"]

            for driver in drivers:
                aux_driver = {}
                if driver in player_stats:                
                    aux_driver = player_stats[driver]
                else:
                    aux_driver = dict([(key,drivers[driver][key]) for key in ["DriverId","FullName","TeamName","HeadshotUrl"]])
                    aux_driver["Wins"] = 0
                    aux_driver["Podiums"] = 0
                    aux_driver["ChampionshipWins"] = 0
                    
                if drivers[driver]["ClassifiedPosition"] == "1":
                    aux_driver["Wins"] = aux_driver["Wins"] + 1

                if drivers[driver]["ClassifiedPosition"] in ["1","2","3"]:
                    aux_driver["Podiums"] = aux_driver["Podiums"] + 1

                player_stats[driver] = aux_driver

                if driver in player_points:
                    player_points[driver] = player_points[driver] + drivers[driver]["Points"] 
                else:
                    player_points[driver] = drivers[driver]["Points"]

        sorted_points = [(key,player_points[key]) for key in player_points]
        sorted_points.sort(key = lambda x: x[1])

        player_stats[sorted_points[0][0]]["ChampionshipWins"] = player_stats[sorted_points[0][0]]["ChampionshipWins"] + 1
        player_stats[sorted_points[1][0]]["ChampionshipWins"] = player_stats[sorted_points[1][0]]["ChampionshipWins"] + 1
        player_stats[sorted_points[2][0]]["ChampionshipWins"] = player_stats[sorted_points[2][0]]["ChampionshipWins"] + 1

        if flag:
            flag = not flag
            for player in player_points:
                player_stats[player]["Points"] = player_points[player]

    print(player_stats)


#conn = mysql.connector.connect(
#    host="svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com",      # Change to your MySQL server host
#    user="gonc",  # Your MySQL username
#    password="cAFjKuRtxj1EpZU4Ka3UHKrMEVqxybAQ",  # Your MySQL password
#    database="f1hub",  # Name of the database
#    port="3333"
#)
#
## Create a cursor object
#cursor = conn.cursor()
#
#player_insert = '''INSERT INTO Player (PlayerId,Name,Team,Image,Wins,Podiums,Points,ChampionshipWins,PlayerNum)
#                   Values ('{PlayerId}','{Name}','{Team}','{Image}',{Wins},{Podiums},{Points},{ChampionshipWins},{PlayerNum});'''
#print(player_insert.format(PlayerId = driver.DriverId,
#                                    Name = driver.FullName,
#                                    Team = driver.TeamName,
#                                    Image = driver.HeadshotUrl,
#                                    Wins = 0,
#                                    Podiums = 0,
#                                    Points = 0,
#                                    ChampionshipWins = 0,
#                                    PlayerNum = "16"))
#
## Execute a simple query
#cursor.execute(player_insert.format(PlayerId = driver.DriverId,
#                                    Name = driver.FullName,
#                                    Team = driver.TeamName,
#                                    Image = driver.HeadshotUrl,
#                                    Wins = 0,
#                                    Podiums = 0,
#                                    Points = 0,
#                                    ChampionshipWins = 0,
#                                    PlayerNum = "16"))
#
#conn.commit()
#
#
## Close connection
#cursor.close()
#conn.close()

# The rest is just plotting
#fig, ax = plt.subplots()
#ax.plot(t, vCar, label='Fast')
#ax.set_xlabel('Time')
#ax.set_ylabel('Speed [Km/h]')
#ax.set_title('Leclerc is')
#ax.legend()
#plt.show()
