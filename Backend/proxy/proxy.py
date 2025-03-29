from flask import Flask, jsonify
from flask_cors import CORS
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables for sensitive information
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database connection parameters
DB_CONFIG = {
    'user': 'gonc',
    'host': 'svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
    'port': 3333,
    'password': 'cAFjKuRtxj1EpZU4Ka3UHKrMEVqxybAQ',
    'database': 'f1hub',
    'auth_plugin': 'mysql_native_password'
}

def driver_nationality(driver_name):
    """
    Returns the nationality for a given driver name.
    
    Args:
        driver_name (str): The name of the F1 driver
    
    Returns:
        str: The nationality of the driver
    """
    if not driver_name:
        return "Unknown"
    
    nationality_map = {
        "Lewis Hamilton": "British",
        "Max Verstappen": "Dutch",
        "Charles Leclerc": "Monegasque",
        "Sebastian Vettel": "German",
        "Fernando Alonso": "Spanish",
        "Lando Norris": "British",
        "Carlos Sainz": "Spanish",
        "Oscar Piastri": "Australian",
        "Yuki Tsunoda": "Japanese",
        "George Russell": "British",
        "Jack Doohan": "Australian",
        "Gabriel Bortoleto": "Brazilian",
        "Liam Lawson": "New Zealander",
        "Pierre Gasly": "French",
        "Esteban Ocon": "French",
        "Lance Stroll": "Canadian",
        "Oliver Bearman": "British",
        "Isack Hadjar": "French",
        "Andrea Kimi Antonelli": "Italian",
        "Kimi Antonelli": "Italian",
        "Nico Hülkenberg": "German",
        "Nico Hulkenberg": "German",
        "Sergio Pérez": "Mexican",
        "Sergio Perez": "Mexican",
        "Valtteri Bottas": "Finnish",
        "Alexander Albon": "Thai",
        "Logan Sargeant": "American",
        "Kevin Magnussen": "Danish",
        "Guanyu Zhou": "Chinese",
        "Franco Colapinto": "Argentinian",
        "Daniel Ricciardo": "Australian"
    }
    
    return nationality_map.get(driver_name, "Unknown")


def get_nationality_code(nationality):
    """
    Returns the country code for a given nationality.
    
    Args:
        nationality (str): The nationality
    
    Returns:
        str: The two-letter country code
    """
    if not nationality:
        return "unknown"
    
    code_map = {
        "Dutch": "nl",
        "British": "gb",
        "Monegasque": "mc",
        "Spanish": "es",
        "Australian": "au",
        "Mexican": "mx",
        "Finnish": "fi",
        "Canadian": "ca",
        "Thai": "th",
        "Japanese": "jp",
        "French": "fr",
        "Danish": "dk",
        "German": "de",
        "American": "us",
        "Chinese": "cn",
        "Italian": "it",
        "Brazilian": "br",
        "New Zealander": "nz",
        "Argentinian": "ar"
    }
    
    return code_map.get(nationality, "unknown")


def get_team_color(team_name):
    """
    Get default team color for a given team name.
    
    Args:
        team_name (str): The name of the F1 team
    
    Returns:
        str: The hex color code for the team
    """
    if not team_name:
        return "#666666"
    
    color_map = {
        "Red Bull Racing": "#0600EF",
        "Mercedes": "#00D2BE",
        "Ferrari": "#DC0000",
        "McLaren": "#FF8700",
        "Aston Martin": "#006F62",
        "Alpine": "#0090FF",
        "Williams": "#005AFF",
        "RB": "#1E41FF",
        "Racing Bulls": "#1E41FF",
        "VCARB": "#1E41FF",
        "Sauber": "#900000",
        "Haas F1 Team": "#FFFFFF"
    }
    
    return color_map.get(team_name, "#666666")

def get_flag_url(nationality):
    """
    Returns the F1 official website flag URL for a given nationality.
    
    Args:
        nationality (str): The nationality of the driver
    
    Returns:
        str: The URL to the nationality flag image from F1's website
    """
    if not nationality:
        return ""
    
    # Base URL for all F1 website flags
    base_url = "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/"
    
    # Map nationalities to their corresponding flag file names
    flags_map = {
        "Dutch": "netherlands-flag.png",
        "British": "great-britain-flag.png",
        "Monegasque": "monaco-flag.png",
        "Spanish": "spain-flag.png",
        "Australian": "australia-flag.png",
        "Mexican": "mexico-flag.png",
        "Finnish": "finland-flag.png",
        "Canadian": "canada-flag.png",
        "Thai": "thailand-flag.png",
        "Japanese": "japan-flag.png",
        "French": "france-flag.png",
        "Danish": "denmark-flag.png",
        "German": "germany-flag.png",
        "American": "united-states-flag.png",
        "Chinese": "china-flag.png",
        "Italian": "italy-flag.png",
        "Brazilian": "brazil-flag.png",
        "New Zealander": "new-zealand-flag.png",
        "Argentinian": "argentina-flag.png",
        "Belgian": "belgium-flag.png",
        "Swiss": "switzerland-flag.png",
        "Russian": "russia-flag.png",
        "Polish": "poland-flag.png",
        "Swedish": "sweden-flag.png",
        "Estonian": "estonia-flag.png",
        "Portuguese": "portugal-flag.png",
        "Indian": "india-flag.png",
        "Austrian": "austria-flag.png",
        "Bahraini": "bahrain-flag.png",
        "Saudi Arabian": "saudi-arabia-flag.png",
        "Emirati": "uae-flag.png",
        "Qatari": "qatar-flag.png",
        "Singaporean": "singapore-flag.png",
        "Hungarian": "hungary-flag.png",
        "Vietnamese": "vietnam-flag.png",
        "Azerbaijani": "azerbaijan-flag.png",
        "South African": "south-africa-flag.png",
        "Malaysian": "malaysia-flag.png",
        "Turkish": "turkey-flag.png",
        "Korean": "south-korea-flag.png",
        "Indonesian": "indonesia-flag.png"
    }
    
    # Get the flag filename from the map or use a default
    flag_filename = flags_map.get(nationality, "")
    
    # If we don't have a mapping for this nationality, return empty string
    if not flag_filename:
        return ""
        
    # Construct and return the full URL
    return base_url + flag_filename


# You can also add a convenience function that combines nationality lookup with flag URL
def get_driver_flag(driver_name):
    """
    Returns the flag URL for a given driver name.
    
    Args:
        driver_name (str): The name of the F1 driver
    
    Returns:
        str: The URL to the driver's nationality flag image
    """
    nationality = driver_nationality(driver_name)
    return get_flag_url(nationality)

def get_db_connection():
    """Establish and return a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

@app.route('/api/standings/drivers', methods=['GET'])
def get_drivers_standings():
    """Retrieve and return driver standings data"""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Query to fetch all driver data
        query = """
        SELECT 
            PlayerId as id,
            Name as name,
            Team as team,
            Image as image,
            Wins as wins,
            Podiums as podiums,
            Points as points,
            ChampionshipWins as championships,
            PlayerNum as number
        FROM Player
        ORDER BY Points DESC
        """
        
        cursor.execute(query)
        drivers = cursor.fetchall()

        print(f"Fetched {len(drivers)} drivers from the database.")
        
        # Add position based on the sorted order and slugify name for URLs
        for i, driver in enumerate(drivers):
            driver['position'] = i + 1
            driver['slug'] = driver['name'].lower().replace(' ', '-')
            
            # Set default values for required frontend fields
            driver['nationality'] = driver_nationality(driver['name'])
            driver['flag'] = get_driver_flag(driver['name'])
            driver['teamColor'] = get_team_color(driver['team'])
        
        return jsonify(drivers)
    
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/api/standings/constructors', methods=['GET'])
def get_constructors_standings():
    """Retrieve and return constructor standings data from driver data"""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Query to group drivers by team and sum their points
        query = """
        SELECT 
            Team as name,
            SUM(Points) as points,
            SUM(Wins) as wins
        FROM Player
        GROUP BY Team
        ORDER BY points DESC
        """
        
        cursor.execute(query)
        constructors = cursor.fetchall()
        
        # Add position and ID based on the sorted order
        for i, team in enumerate(constructors):
            team['id'] = i + 1
            team['position'] = i + 1
            
            # Set default values for required fields
            team['color'] = get_team_color(team['name'])
            team['logo'] = ""
            team['country'] = driver_nationality(team['name'])
            team['flag'] = ""
        
        return jsonify(constructors)
    
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

#get a driver by ID
@app.route('/api/driver/<int:driver_id>', methods=['GET'])
def get_driver_by_id(driver_id):
    """Retrieve and return driver data by ID"""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Query to fetch driver data by ID
        query = """
        SELECT 
            PlayerId as id,
            Name as name,
            Team as team,
            Image as image,
            Wins as wins,
            Podiums as podiums,
            Points as points,
            ChampionshipWins as championships,
            PlayerNum as number
        FROM Player
        WHERE PlayerId = %s
        """
        
        cursor.execute(query, (driver_id,))
        driver = cursor.fetchone()
        
        if driver:
            driver['slug'] = driver['name'].lower().replace(' ', '-')
            return jsonify(driver)
        else:
            return jsonify({"error": "Driver not found"}), 404
    
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Add this new endpoint after your existing endpoints
@app.route('/api/driver/slug/<string:driver_slug>', methods=['GET'])
def get_driver_by_slug(driver_slug):
    """Retrieve and return driver data by slug"""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Query to fetch all drivers
        query = """
        SELECT 
            PlayerId as id,
            Name as name,
            Team as team,
            Image as image,
            Wins as wins,
            Podiums as podiums,
            Points as points,
            ChampionshipWins as championships,
            PlayerNum as number
        FROM Player
        """
        
        cursor.execute(query)
        drivers = cursor.fetchall()
        
        # Find driver with matching slug
        for driver in drivers:
            # Generate slug from name
            slug = driver['name'].lower().replace(' ', '-')
            if slug == driver_slug:
                driver['slug'] = slug
                driver['nationality'] = "Unknown"  # This would come from your DB in a real app
                driver['flag'] = ""  # This would come from your DB in a real app
                return jsonify(driver)
        
        # If we get here, no driver was found
        return jsonify({"error": "Driver not found"}), 404
    
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Use environment variables for production, defaults for development
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


#"get nationality hardcoded for now"


