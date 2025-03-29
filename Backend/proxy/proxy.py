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
            driver['nationality'] = "Unknown"
            driver['flag'] = ""
            driver['teamColor'] = "#666666"  # Default color if no mapping is available
        
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
            team['color'] = "#666666"
            team['logo'] = ""
            team['country'] = "Unknown"
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