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
    PlayerNum INT NOT NULL
);

-- Creating the Event table to store events associated with players
-- CREATE TABLE Event (
--     EventId INT AUTO_INCREMENT PRIMARY KEY,
--     PlayerId INT,
--     EventDescription TEXT NOT NULL,
--     EventDate DATETIME DEFAULT CURRENT_TIMESTAMP,
--     FOREIGN KEY (PlayerId) REFERENCES Player(PlayerId) ON DELETE CASCADE
-- );

