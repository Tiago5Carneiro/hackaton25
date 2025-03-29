CREATE DATABASE IF NOT EXISTS f1hub;
USE f1hub;

-- Creating the Player table
CREATE TABLE Player (
    PlayerId VARCHAR(100) PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Team VARCHAR(100) NOT NULL,
    Image VARCHAR(200) NOT NULL,
    Wins INT NOT NULL,
    Podiums INT NOT NULL,
    Points INT NOT NULL,
    ChampionshipWins INT NOT NULL,
    PlayerNum INT NOT NULL,
    IsActive BOOLEAN NOT NULL
);

-- Creating the Event table to store events associated with players
 CREATE TABLE Event (
    EventName VARCHAR(200) NOT NULL,
    StartDate DATETIME NOT NULL PRIMARY KEY,
    EndDate DATETIME NOT NULL,
    Country VARCHAR(100) NOT NULL,
    Location VARCHAR(100) NOT NULL,
    Winner VARCHAR(100) NOT NULL,
    FastLap INT NOT NULL
 );

-- Junction table for many-to-many relationship
CREATE TABLE PlayerEvent (
    PlayerId VARCHAR(100),
    EventId DATETIME,
    PRIMARY KEY (PlayerId, EventId)
);
