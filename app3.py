from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, emit #
from datetime import datetime
from haversine import haversine
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import defaultdict
import requests  # Make sure you have imported the requests module
from gtts import gTTS
import os

from flask_sqlalchemy import SQLAlchemy
import sqlite3

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///queue2_system.db'
db = SQLAlchemy(app)
app.secret_key = 'your_secret_key'  # Necessary for session managementpi
socketio = SocketIO(app)
CORS(app)

# Hardcoded venue location for location-based queue joining
venue_lat = -1.2640637
venue_lon = 36.9077355

office_codes = {
    'Health Services': 'HS',
    'Identification and Childhood Services': 'IC',
    'Legal,Crime and Oversite ': 'LC',
    'Education Services': 'ES',
    'Business and Commerce': 'BC'
}
 #Counter for ticket numbers
office_counters = {
    'HS': 1,
    'IC': 1,
    'LC': 1,
    'ES': 1,
    'BC': 1
}

def create_connection():
    """Create a SQLite database connection."""
    try:
        conn = sqlite3.connect('queue2_system.db')
        return conn
    except sqlite3.Error as e:
        print(f"Error: {e}")
    return None

# Function to create tables if they don't exist
def create_tables():
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()

        # Create the users2 table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                id_no TEXT,
                registration_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queues2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT,
    ticket_number TEXT,
    booth TEXT,
    processed INTEGER DEFAULT 0,  -- Boolean column (0 = False, 1 = True)
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users2(id)
    

);
                       ''')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS served2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT,
                ticket_number TEXT,
                booth TEXT,
                served_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users2(id)
            )
        ''')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS staff (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      office TEXT NOT NULL,
                      email TEXT NOT NULL UNIQUE,
                  password TEXT NOT NULL,
                agreed_terms BOOLEAN NOT NULL DEFAULT 0
);

        ''')
    cursor.execute('''
               CREATE TABLE IF NOT EXISTS staff_login (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (email) REFERENCES staff(email)
);
''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS skipped (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT NOT NULL,
    ticket_number TEXT NOT NULL,
    booth TEXT,
    skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users2(id)
);

      ''')
       
    conn.commit()
    conn.close()
    
    print("Error! Cannot create the database connection.")

# Initialize tables when the app starts
create_tables()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_button_click', methods=['POST'])
def handle_button_click():
    action = request.form['action']
    if action == 'register':
        return redirect(url_for('register'))
    elif action == 'join_remotely':
        return redirect(url_for('join_remotely'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        id_no = request.form['id_no']
        registration_time = datetime.utcnow()

        conn = create_connection()
        if conn is None:
            flash("Database connection failed!", "error")
            return redirect(url_for('register'))

        try:
            cursor = conn.cursor()
            # Insert the registration data
            insert_query = """
                INSERT INTO users2 (name, id_no, registration_time)
                VALUES (?, ?, ?)
            """
            cursor.execute(insert_query, (name, id_no, registration_time))
            conn.commit()

            flash("Registration successful!", "success")
            return redirect(url_for('dashboard'))

        except sqlite3.Error as e:
            flash(f"Error while inserting data: {e}", "error")
            conn.rollback()
            return redirect(url_for('register'))

        finally:
            cursor.close()
            conn.close()

    return render_template('Registration.html')

@app.route('/join_remotely')
def join_remotely():
    return render_template('join_remotely.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', office_codes=office_codes)
@app.route('/generate_ticket', methods=['POST'])
def generate_ticket():
    office_code = request.form['office_code']
    ticket_number = f"{office_code}{office_counters[office_code]:03d}"
    office_counters[office_code] += 1

    try:
        conn = create_connection()
        cursor = conn.cursor()

        # Fetch the most recent user
        cursor.execute("SELECT id, name FROM users2 ORDER BY id DESC LIMIT 1")
        user = cursor.fetchone()
        created_at = "Ticket not created"
        booth = "Booth not found"

        if user:
            user_id, name = user

            # Map office code to booth
            ticket_code_to_room = {
                'HS': 'Booth 1',
                'IC': 'Booth 2',
                'LC': 'Booth 3',
                'ES': 'Booth 4',
                'BC': 'Booth 5'
            }
            room_number = ticket_code_to_room.get(office_code, 'Default Room')

            # Insert the ticket into queues2
            cursor.execute(
                "INSERT INTO queues2 (user_id, name, ticket_number, booth) VALUES (?, ?, ?, ?)", 
                (user_id, name, ticket_number, room_number)
            )
            conn.commit()

            # Fetch the created_at timestamp
            cursor.execute(
                "SELECT processed_at FROM queues2 WHERE ticket_number = ? ORDER BY id DESC LIMIT 1", 
                (ticket_number,)
            )
            result = cursor.fetchone()
            if result:
                created_at = result[0]
            booth = room_number

        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error: {e}")

    return render_template('tickets.html', ticket_number=ticket_number, booth=booth, created_at=created_at)
@app.route('/ticket_status/<string:ticket_number>')
def ticket_status(ticket_number):
    try:
        conn = create_connection()
        cursor = conn.cursor()

        # Fetch the position of the ticket in the queue
        cursor.execute(
            """
            SELECT COUNT(*) FROM queues2 
            WHERE ticket_number < ? AND processed = 0 AND ticket_number LIKE ?
            """, 
            (ticket_number, ticket_number[:2] + '%')
        )
        position = cursor.fetchone()[0]

        # Fetch the current unprocessed ticket
        cursor.execute(
            """
            SELECT id, user_id, name, ticket_number, booth, processed_at 
            FROM queues2 
            WHERE processed = 0 
            ORDER BY processed_at LIMIT 1
            """
        )
        current_ticket = cursor.fetchone()

        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error: {e}")
        position = None
        current_ticket = None

    return render_template(
        'ticket-status.html', 
        ticket_number=ticket_number, 
        current_ticket=current_ticket, 
        position=position
    )

@app.route('/register_staff', methods=['GET', 'POST'])
def register_staff():
    if request.method == 'POST':
        name = request.form['Name']
        office = request.form['office']
        email = request.form['email']
        password = request.form['password']
        agreed_terms = 'iAgree' in request.form

        hashed_password = generate_password_hash(password)  # Hash the password

        conn = create_connection()
        if conn is None:
            flash("Database connection failed!", "error")
            return redirect(url_for('register_staff'))

        try:
            cursor = conn.cursor()

            insert_query = """
            INSERT INTO staff (name, office, email, password, agreed_terms)
            VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (name, office, email, hashed_password, agreed_terms))
            conn.commit()

            flash("Staff registration successful!", "success")

            return redirect(url_for('staff_dashboard'))  # Redirect to staff_dashboard after registration

        except sqlite3.Error as e:
            flash(f"Error while inserting data: {e}", "error")
            conn.rollback()
            return redirect(url_for('register_staff'))

        finally:
            cursor.close()
            conn.close()

    return render_template('staff-registration.html')


@app.route('/staff_dashboard')
def staff_dashboard():
    # Directly query database for tickets (no session check)
    conn = create_connection()
    cursor = conn.cursor()

    
        # Query database for tickets
    cursor.execute("SELECT * FROM queues2")
    tickets = cursor.fetchall()

        # Fetch the count for people in queue, served, and skipped
    queue_count = get_count_from_table('queues2')  # For the queues table
    served_count = get_count_from_table('served2')  # For the served table
    skipped_count = get_count_from_table('skipped')  # For the skipped table

    cursor.close()
    conn.close()

        # Pass the counts and tickets data to the template
    return render_template('staff-dashboard.html', tickets=tickets, queue_count=queue_count, served_count=served_count, skipped_count=skipped_count)

   

# Helper function to get the count of rows in a table
def get_count_from_table(table_name):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count
@app.route('/done/<int:id>', methods=['POST'])
def done_ticket(id):
    conn = create_connection()
    cursor = conn.cursor()

    try:
        # Fetch the ticket details
        cursor.execute("SELECT user_id, name, ticket_number, booth FROM queues2 WHERE id = ?", (id,))
        ticket = cursor.fetchone()

        if ticket:
            # Update the ticket status to processed in queues2
            cursor.execute("UPDATE queues2 SET processed = 1 WHERE id = ?", (id,))

            # Move the processed ticket to the served2 table
            cursor.execute("""
                INSERT INTO served2 (user_id, name, ticket_number, booth)
                VALUES (?, ?, ?, ?)
            """, ticket)

            # Delete the processed ticket from queues2
            cursor.execute("DELETE FROM queues2 WHERE id = ?", (id,))
            conn.commit()

            flash("Ticket processed successfully.", "success")
    except sqlite3.Error as e:
        conn.rollback()
        flash(f"Error processing ticket: {e}", "error")
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('queue_screen'))

@app.route('/skip/<int:id>', methods=['POST'])
def skip_ticket(id):
    conn = create_connection()
    cursor = conn.cursor()

    try:
        # Fetch the ticket details
        cursor.execute("SELECT user_id, name, ticket_number, booth FROM queues2 WHERE id = ?", (id,))
        ticket = cursor.fetchone()

        if ticket:
            # Update the ticket status to processed in queues2
            cursor.execute("UPDATE queues2 SET processed = 1 WHERE id = ?", (id,))

            # Move the ticket to the skipped table
            cursor.execute("""
                INSERT INTO skipped (user_id, name, ticket_number, booth)
                VALUES (?, ?, ?, ?)
            """, ticket)

            # Delete the ticket from queues2
            cursor.execute("DELETE FROM queues2 WHERE id = ?", (id,))
            conn.commit()

            flash("Ticket skipped successfully.", "success")
    except sqlite3.Error as e:
        conn.rollback()
        flash(f"Error skipping ticket: {e}", "error")
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('queue_screen'))

@app.route('/skipped')
def skipped_list():
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all records from the skipped table, ordered by the skipped_at timestamp
    cursor.execute("SELECT id, ticket_number, user_id, name,booth, skipped_at FROM skipped ORDER BY skipped_at DESC")
    skipped_records = cursor.fetchall()
    return render_template('skipped.html', skipped_records=skipped_records)  
@app.route('/transfer_back/<int:id>', methods=['POST'])
def transfer_back(id):
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch the skipped ticket details
    cursor.execute("SELECT ticket_number, user_id FROM skipped WHERE id = ?", (id,))
    skipped_ticket = cursor.fetchone()

    if skipped_ticket:
        # Insert the ticket back into the queue
        cursor.execute("INSERT INTO queues2 (ticket_number, user_id, processed) VALUES (?, ?, ?)", 
                       (skipped_ticket[0], skipped_ticket[1], False))

        # Remove the ticket from the skipped table
        cursor.execute("DELETE FROM skipped WHERE id = ?", (id,))
        conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('skipped_list'))

@app.route('/remove_skipped/<int:id>', methods=['POST'])
def remove_skipped(id):
    conn = create_connection()
    cursor = conn.cursor()

    # Remove the ticket from the skipped table
    cursor.execute("DELETE FROM skipped WHERE id = ?", (id,))
    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('skipped_list'))
@app.route('/view_queues')
def view_queues():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM queues2")
    queues2 = cursor.fetchall()
    return render_template('queue-list.html', queues2=queues2)
@app.route('/queue-screen', methods=['GET', 'POST'])
def queue_screen():
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch the current ticket (the first unprocessed one)
    cursor.execute("SELECT id, ticket_number FROM queues2 WHERE processed = 0 ORDER BY id LIMIT 1")
    current_ticket = cursor.fetchone()

    # Fetch the next ticket in the queue
    if current_ticket:
        cursor.execute("SELECT ticket_number FROM queues2 WHERE processed = 0 AND id > ? ORDER BY id LIMIT 1", (current_ticket[0],))
        next_ticket = cursor.fetchone()
    else:
        next_ticket = None

    cursor.close()
    conn.close()

    return render_template('queue-screen.html', current_ticket=current_ticket, next_ticket=next_ticket)

@app.route('/process_ticket', methods=['POST'])
def process_ticket():
    try:
        conn = create_connection()
        cursor = conn.cursor()

        # Fetch the first unprocessed ticket from the queues2 table
        cursor.execute(
            "SELECT id, user_id, name, ticket_number,booth FROM queues2 WHERE processed = FALSE ORDER BY id ASC LIMIT 1"
        )
        current_ticket = cursor.fetchone()

        if current_ticket:
            ticket_id = current_ticket[0]
            user_id = current_ticket[1]
            name = current_ticket[2]
            ticket_number = current_ticket[3]
            booth = current_ticket[4]

            # Mark the ticket as processed by setting processed to TRUE
            cursor.execute("UPDATE queues2 SET processed = TRUE WHERE id = ?", (ticket_id,))

            # Move the processed ticket to the served2 table
            cursor.execute(
                "INSERT INTO served2 (user_id, name, ticket_number,booth, served_at) "
                "VALUES (?, ?, ?, ?, ?, NOW())", 
                (user_id, name, ticket_number, booth)
            )

            # Delete the processed ticket from queues2
            cursor.execute("DELETE FROM queues2 WHERE id = ?", (ticket_id,))

            # Commit the changes to the database
            conn.commit()

            # Notify clients about the queue update (if using Socket.IO)
            socketio.emit('update_queue')

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        flash("An error occurred while processing the ticket.", "error")
    
    return redirect(url_for('queue_screen'))

@app.route('/graphs')
def generate_graphs():
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch data for people who visited (from users2 table)
    cursor.execute("SELECT strftime('%w', registration_time), COUNT(*) FROM users2 GROUP BY strftime('%w', registration_time)")
    visitors_by_day = defaultdict(int, cursor.fetchall())

    # Fetch data for people who were served (from served2 table)
    cursor.execute("SELECT strftime('%w', served_at), COUNT(*) FROM served2 GROUP BY strftime('%w', served_at)")
    served_by_day = defaultdict(int, cursor.fetchall())

    # Fetch data for people who were skipped (from skipped table)
    cursor.execute("SELECT strftime('%w', skipped_at), COUNT(*) FROM skipped GROUP BY strftime('%w', skipped_at)")
    skipped_by_day = defaultdict(int, cursor.fetchall())

    cursor.close()
    conn.close()

    # Create the graphs
    img1, img2, img3 = generate_graph(visitors_by_day, served_by_day, skipped_by_day)

    # Render the reports.html page with the graphs
    return render_template('report.html', img1=img1, img2=img2, img3=img3)

def generate_graph(visitors_by_day, served_by_day, skipped_by_day):
    days_of_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    # Convert defaultdicts to lists for plotting
    visitors = [visitors_by_day[str(i)] for i in range(7)]  # Ensure str keys
    served = [served_by_day[str(i)] for i in range(7)]
    skipped = [skipped_by_day[str(i)] for i in range(7)]

    # Graph 1: People who visited
    plt.figure()
    plt.bar(days_of_week, visitors, color='blue')
    plt.title('People Who Visited Per Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    img1 = get_graph()

    # Graph 2: People who were served
    plt.figure()
    plt.bar(days_of_week, served, color='green')
    plt.title('People Who Were Served Per Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    img2 = get_graph()

    # Graph 3: People who were skipped
    plt.figure()
    plt.bar(days_of_week, skipped, color='red')
    plt.title('People Who Were Skipped Per Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    img3 = get_graph()

    return img1, img2, img3

def get_graph():
    # Convert the plot to a PNG image and encode it to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()  # Close the plot to free memory
    return img
@socketio.on('request_queue_update')
def handle_queue_update(office_code):
    data = {
        'current_ticket': 'Some ticket data',
        'people_in_queue': 42
    }
    emit('queue_status_update', data)

@socketio.on('request_user_position')
def handle_user_position(data):
    ticket_number = data['ticket_number']
    office_code = data['office_code']
    data = {
        'people_ahead': 5
    }
    emit('user_position_update', data)
@socketio.on('connect')
def handle_connect():
    print('Client connected')
@app.route('/call_queue', methods=['POST'])
def call_queue():
    queue_id = request.form['queue_id']
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT ticket_number, booth FROM queues2 WHERE id = ?", (queue_id,))
    ticket = cursor.fetchone()

    ticket_number, booth = ticket if ticket else (None, None)
    cursor.close()
    conn.close()

    announcement_text = f"Ticket {ticket_number}, please proceed to {booth}."
    tts = gTTS(announcement_text)
    tts.save("announcement.mp3")
    os.system("start announcement.mp3")
    return redirect(url_for('queue_screen'))

    

if __name__ == '__main__':
    socketio.run(app, debug=True)