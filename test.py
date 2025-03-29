import datetime
import fastf1 
import pandas as pd


events = None
races = None


def get_total_events():
    global events
    events = []
    current_date = datetime.datetime.now()
    current_year = current_date.year
    for x in range(2019,current_year):
        print(x)
        start = datetime.datetime(x,1,1)

        event_schedule = fastf1.get_events_remaining(start)

        event_name = event_schedule['EventName'].tolist()
        event_date = pd.to_datetime(event_schedule['EventDate'])
        
        for event,date in zip(event_name,event_date):
            if x == current_year:
                if not current_date < date: 
                    print(f"Current event: {event}")
                    events.append(fastf1.get_event(x,event))
            else : 
                events.append(fastf1.get_event(x,event))

def get_races():
    global events,races
    races = []
    for event in events:
        races.append(event.get_race())
    print(f"Total races: {len(races)}")

def load_races():
    for race in races :
        race.load(messages = False)




def main():
    global events,token
    print(f"Total events: {len(events)}")

if __name__ == "__main__":
    get_total_events()
    get_races()
    main()