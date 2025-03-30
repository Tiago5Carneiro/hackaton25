import fastf1
import pandas as pd
import numpy as np
import mysql.connector 
import datetime
import time
import stat_prediction


conn = mysql.connector.connect(
    host="svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com",      # Change to your MySQL server host
    user="gonc",  # Your MySQL username
    password="cAFjKuRtxj1EpZU4Ka3UHKrMEVqxybAQ",  # Your MySQL password
    database="f1hub",  # Name of the database
    port="3333"
)

def get_future_events():
    start = datetime.datetime.now()

    event_schedule = fastf1.get_events_remaining(start, include_testing= False)
    event_schedule = event_schedule[event_schedule.EventFormat == "conventional"]

    for _, event in event_schedule.iterrows():
        cursor = conn.cursor()

        insert = '''INSERT INTO FutureEvent (EventName, Date, Country, Location) 
                    VALUES (%s,%s,%s,%s)'''

        cursor.execute(insert, (
            event["EventName"],
            event["EventDate"],
            event["Country"],
            event["Location"]
        ))

        conn.commit()
        cursor.close()





def get_year_events(year):
    events = []
    
    start = datetime.datetime(year,1,1)

    event_schedule = fastf1.get_events_remaining(start, include_testing= False)

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

def get_qualifying(events):
    qualifyings = []
    current_date = datetime.datetime.now()
    for event in events:
        if event.EventDate < current_date:
            qualifyings.append(event.get_qualifying())
    return qualifyings

def get_races(events):
    races = []
    current_date = datetime.datetime.now()
    for event in events:
        if event.EventDate < current_date:
            races.append(event.get_race())
    return races

def get_sprints(events):
    sprints = []
    current_date = datetime.datetime.now()
    for event in events:
        if event.EventDate < current_date:
            try:
                sprints.append(event.get_sprint())
            except:
                pass
    return sprints 

def get_stats_qualifyings(session):
    session.load(messages = False)

    ret = {}

    for i in session.drivers:
        results = session.results
        results = results[results.DriverNumber == i]

        if not pd.isnull(results.Q1).item():
            ret[i] =  0.2
        elif not pd.isnull(results.Q2).item():
            ret[i] =  0.5
        elif not pd.isnull(results.Q3).item():
            ret[i] =  1

    return ret

def get_stats_race(session):
    print(session)
    session.load(messages = False)
    
    laps = session.laps
    car_data = session.car_data
    n_lap = 1 if session.total_laps == 0 else session.total_laps 
    
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
        if not np.isnan(results[results.DriverNumber == i].Points.iloc[0]):
            driver["Points"] = int(results[results.DriverNumber == i].Points.iloc[0])
        else:
            driver["Points"] = 0 

        driver["Stint"] = 0
    
        laps_info = laps[laps.Driver == driver.Abbreviation]
        car_info = car_data[i]
        #car_info = car_info.drop([x for x in range(car_info.shape[0]) if x % 5 == 0])
    
        last_lap = session.session_start_time
        last_position = driver.GridPosition
        last_stint = 0
        stats_laps = []
        #falta detetar se se espetou
        # verificar qualidade de dados para filtrar
        event_speed_average = 0
        event_delta_position = 0
        for _,lap in laps_info.iterrows():
            stats_lap = {}
    
            car_info_lap = car_info[np.logical_and(last_lap < car_info.Time, car_info.Time < lap.Time)] 
            mean_speed = int(car_info_lap.Speed.mean())
            event_speed_average = event_speed_average + mean_speed 
            event_delta_position = event_delta_position + lap.Position - last_position
    
            stats_lap["Speed"] = mean_speed
            stats_lap["CurrentPosition"] = lap.Position
            stats_lap["DeltaPosition"] = lap.Position - last_position

            if last_stint != lap.Stint:
                last_stint = lap.Stint
                stint = 0
                if lap.Compound in ["HARD","WET"]:
                    stint = 3
                elif lap.Compound in ["MEDIUM","INTERMEDIATE"]:
                    stint = 1.5
                else:
                    stint = 1
                driver["Stint"] = driver["Stint"] + stint
    
            stats_laps.append(stats_lap)
    
            last_lap = lap.Time
            last_position = lap.Position
    
        driver["AverageSpeed"] = event_speed_average/n_lap
        driver["AverageDeltaPosition"] = 0 if np.isnan(event_delta_position/n_lap) else event_delta_position/n_lap
        driver["Lap"] = stats_laps
        drivers[i] = driver
    
    event_info["Drivers"] = drivers

    return event_info


def add_event_bd(event_info):
    cursor = conn.cursor()

    # Insert the event
    event_insert = '''
        INSERT INTO Event (EventName, StartDate, EndDate, Country, Location, Winner, FastLap)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    '''
    
    winner = event_info["Winner"]["DriverId"]
    fasttime = int(event_info["FastLap"].Time.total_seconds() * 1000)

    cursor.execute(event_insert, (
        event_info["EventName"],
        event_info["StartDate"],
        event_info["EndDate"],
        event_info["Country"],
        event_info["Location"],
        winner,
        fasttime
    ))

    # Insert into PlayerEvent table
    player_event_insert = '''
        INSERT INTO PlayerEvent (PlayerId, StartDate,Location) VALUES (%s, %s, %s)
    '''
    
    cursor.execute(player_event_insert, (winner, event_info["StartDate"],event_info["Location"]))

    # Commit changes
    conn.commit()

    # Close cursor and connection
    cursor.close()


def add_card_driver_bd(driver,card):

    cursor = conn.cursor()

    insert = ''' UPDATE Player SET %s = %s, %s = %s, %s = %s, %s = %s, %s = %s, %s = %s, %s = %s 
                 WHERE PlayerId = %s;
             '''

    tup = ()
    for info in card:
        tup = tup + (info,card[info])

    tup = tup + driver 

    cursor.execute(insert,driver)

    conn.commit()

    cursor.close()

def add_driver_bd(driver,driverId):
   
   # Create a cursor object
    cursor = conn.cursor()
   
    player_insert = '''INSERT INTO Player (PlayerId,Name,Team,Image,Wins,Podiums,Points,ChampionshipWins,PlayerNum,IsActive)
                       Values ('{PlayerId}','{Name}','{Team}','{Image}',{Wins},{Podiums},{Points},{ChampionshipWins},{PlayerNum},{IsActive})
                       ON DUPLICATE KEY UPDATE 
                           Wins = Wins + VALUES(Wins),
                           Podiums = Podiums + VALUES(Podiums),
                           ChampionshipWins = ChampionshipWins + VALUES(ChampionshipWins);'''


    # Execute a simple query
    cursor.execute(player_insert.format(PlayerId = driver["DriverId"],
                                        Name = driver["FullName"],
                                        Team = driver["TeamName"],
                                        Image = driver["HeadshotUrl"],
                                        Wins = driver["Wins"],
                                        Podiums = driver["Podiums"],
                                        Points = driver["Points"],
                                        ChampionshipWins = driver["ChampionshipWins"],
                                        PlayerNum = driverId,
                                        IsActive = driver["IsActive"]))
    
    conn.commit()
    
    
    # Close connection
    cursor.close()

def update_driver_bd(driverId,field,value):
   
   # Create a cursor object
    cursor = conn.cursor()
   
    update_player = "UPDATE Player SET {field} = {field} + {value} WHERE PlayerId = '{driverId}'"

    print(update_player.format(driverId = driverId,
                                        field = field,
                                        value = value))

    # Execute a simple query
    cursor.execute(update_player.format(driverId = driverId,
                                        field = field,
                                        value = value))
    
    conn.commit()
    
    
    # Close connection
    cursor.close()


if __name__ == "__main__":
    get_future_events()

    current_year = datetime.datetime.now().year

    races = []

    flag = True

    for year in range(current_year,2023,-1):
        events = get_year_events(year)
        races = get_races(events)
        sprints = get_sprints(events)
        qualifyings = get_qualifying(events)

        player_stats = {}

        player_points = {}

        for qualification in qualifyings:
            stats = get_stats_qualifyings(qualification)

            for driver in stats:
                if driver in player_stats:
                    player_stats[driver]["Qualifying"].append(stats[driver])
                else:
                    player_stats[driver] = {}
                    player_stats[driver]["Qualifying"] = [stats[driver]]
                    player_stats[driver]["AvgSpeed"] = []
                    player_stats[driver]["DeltaPos"] = []
                    player_stats[driver]["Consistency"] = []
                    player_stats[driver]["TireMan"] = []

        for race in races:
            stats = get_stats_race(race)

            drivers = stats["Drivers"]

            for driver in drivers:
                aux_driver = {}
                aux_driver = dict([(key,drivers[driver][key]) for key in ["DriverId","FullName","TeamName","HeadshotUrl"]])
                aux_driver["Wins"] = 0
                aux_driver["Podiums"] = 0
                aux_driver["ChampionshipWins"] = 0
                aux_driver["Points"] = 0
                aux_driver["IsActive"] = False
                    
                if drivers[driver]["ClassifiedPosition"] == "1":
                    aux_driver["Wins"] = aux_driver["Wins"] + 1

                if drivers[driver]["ClassifiedPosition"] in ["1","2","3"]:
                    aux_driver["Podiums"] = aux_driver["Podiums"] + 1

                add_driver_bd(aux_driver,driver)

                if driver in player_points:
                    player_points[driver] = (player_points[driver][0] + drivers[driver]["Points"],player_points[driver][1])
                else:
                    player_points[driver] = (drivers[driver]["Points"],aux_driver["DriverId"])

                if driver in player_stats:
                    player_stats[driver]["AvgSpeed"].append(drivers[driver]["AverageSpeed"])
                    player_stats[driver]["DeltaPos"].append(drivers[driver]["AverageDeltaPosition"])
                    player_stats[driver]["Consistency"].append(drivers[driver]["ClassifiedPosition"])
                    player_stats[driver]["TireMan"].append(drivers[driver]["Stint"])
                else:
                    player_stats[driver]["AvgSpeed"] = [drivers[driver]["AverageSpeed"]]
                    player_stats[driver]["DeltaPos"] = [drivers[driver]["AverageDeltaPosition"]]
                    player_stats[driver]["Consistency"] = [drivers[driver]["ClassifiedPosition"]]
                    player_stats[driver]["TireMan"] = [drivers[driver]["Stint"]]

            add_event_bd(stats)

            new_player_stats = {}

            for driver in player_stats:
                new_player_stats[drivers[driver]["DriverId"]] = player_stats[driver]

        max_speed = 0
        max_delta_position = 0
        min_delta_position = 0
        max_stint_position = 0
        for player in player_stats:
            player_dic = player_stats[player]
            max_speed = max(player_dic["AvgSpeed"]) if max(player_dic["AvgSpeed"]) > max_speed else max_speed
            max_delta_position = max(player_dic["DeltaPos"]) if max(player_dic["DeltaPos"]) > max_delta_position else max_delta_position 
            min_delta_position = min(player_dic["DeltaPos"]) if min(player_dic["DeltaPos"]) < min_delta_position else min_delta_position 
            max_stint_position = max(player_dic["TireMan"]) if max(player_dic["TireMan"]) > max_stint_position else max_stint_position 

        for player in player_stats:
            player_dic = player_stats[player]
            player_stats[player]["AvgSpeed"] = [int((x/max_speed) * 50 + 50) for x in player_dic["AvgSpeed"]]
            player_stats[player]["DeltaPos"] = [((x - min_delta_position)/(max_delta_position - min_delta_position)) * 50 + 50 for x in player_dic["DeltaPos"]]
            player_stats[player]["TireMan"] = [(x/max_stint_position) * 50 +50 for x in player_dic["TireMan"]]
            player_stats[player]["Qualifying"] = [x * 50 + 50 for x in player_dic["Qualifying"]]
            l_data = player_stats[player]["Consistency"]
            new_data = []

            for data in l_data:
                try:
                    new_data.append(int(data))
                except:
                    new_data.append(20)

            new_data = [((21 - x)/20) * 50 +50 for x in new_data]
            player_stats[player]["Consistency"] = new_data

            for entry in player_stats[player]:
                player_stats[player][entry] = [int(round(x)) for x in player_stats[player][entry]]


        if year == 2024:
            cards = stat_prediction.card_creation(player_stats)
        
            print(cards)

            for card in cards:
                add_card_driver_bd(card,cards[card])



        sorted_points = [player_points[key] for key in player_points]
        sorted_points.sort(key = lambda x: x[0])

        if flag:
            for sprint in sprints:
                stats = get_stats_race(sprint)
                drivers = stats["Drivers"]

                for driver in drivers:
                    if driver in player_points:
                        player_points[driver] = (player_points[driver][0] + drivers[driver]["Points"],player_points[driver][1])
                    else:
                        player_points[driver] = (drivers[driver]["Points"],driver["DriverId"])

        print(player_points)

        if flag:
            flag = not flag
            for player in player_points:
                update_driver_bd(player_points[player][1],"Points",player_points[player][0])  
                update_driver_bd(player_points[player][1],"IsActive",True)
        else:
            update_driver_bd(sorted_points[0][0],"ChampionshipWins",1)

    conn.close()



# The rest is just plotting
#fig, ax = plt.subplots()
#ax.plot(t, vCar, label='Fast')
#ax.set_xlabel('Time')
#ax.set_ylabel('Speed [Km/h]')
#ax.set_title('Leclerc is')
#ax.legend()
#plt.show()
