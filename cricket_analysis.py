import matplotlib
matplotlib.use('Agg')  # Set backend before other imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np
import matplotlib.patches as patches
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import math
import json
from datetime import datetime
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import copy
from collections import defaultdict
from ttkbootstrap import Style
import os
from collections import defaultdict

def define_game_phases(total_overs):
    powerplay_overs = max(1, round(total_overs * 0.3))
    death_overs = max(1, round(total_overs * 0.25))
    middle_start = powerplay_overs
    middle_end = total_overs - death_overs

    return {
        "Powerplay": (0, powerplay_overs - 1),
        "Middle Overs": (middle_start, middle_end - 1),
        "Death Overs": (middle_end, total_overs - 1)
    }

# ================= Main Application =================
class CricketAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cricket Analysis Manager")
        self.root.geometry("800x600")
        self.style = ttkb.Style(theme='cyborg')
        initialize_database()
        self.create_main_menu()
    
    def create_main_menu(self):
        main_frame = ttkb.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill='both')
        
        ttkb.Label(main_frame, text="Cricket Analysis System", 
                  font=('Helvetica', 16, 'bold'), bootstyle=PRIMARY).pack(pady=20)
        
        menu_buttons = [
            ('Manage Teams', self.manage_teams),
            ('New Match', self.new_match),
            ('Previous Matches', self.view_matches),
            ('Exit', self.root.quit)
        ]
        
        for text, command in menu_buttons:
            ttkb.Button(main_frame, text=text, width=25,
                       command=command, bootstyle=SUCCESS).pack(pady=8)
    
    def manage_teams(self):
        manage_window = ttkb.Toplevel(self.root)
        manage_window.title("Team Management")
        
        notebook = ttkb.Notebook(manage_window)
        
        # Countries Tab
        countries_frame = ttkb.Frame(notebook)
        self.create_country_ui(countries_frame)
        notebook.add(countries_frame, text="Countries")
        
        # Players Tab
        players_frame = ttkb.Frame(notebook)
        self.create_player_ui(players_frame)
        notebook.add(players_frame, text="Players")
        
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
    
    def create_country_ui(self, parent):
        ttkb.Label(parent, text="Add New Country:").grid(row=0, column=0, padx=5, pady=5)
        self.new_country_entry = ttkb.Entry(parent)
        self.new_country_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttkb.Button(parent, text="Add Country", 
                   command=self.add_country, bootstyle=INFO).grid(row=0, column=2, padx=5)
        
        self.country_tree = ttkb.Treeview(parent, columns=('ID', 'Name'), show='headings', bootstyle=PRIMARY)
        self.country_tree.heading('ID', text='ID')
        self.country_tree.heading('Name', text='Country Name')
        self.country_tree.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        self.load_countries()
    
    def add_country(self):
        country_name = self.new_country_entry.get()
        if country_name:
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            try:
                c.execute("INSERT INTO countries (country_name) VALUES (?)", (country_name,))
                conn.commit()
                self.load_countries()
                self.new_country_entry.delete(0, tk.END)
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Country already exists!")
            conn.close()
    
    def load_countries(self):
        self.country_tree.delete(*self.country_tree.get_children())
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT * FROM countries")
        for row in c.fetchall():
            self.country_tree.insert('', 'end', values=row)
        conn.close()
    
    def create_player_ui(self, parent):
        ttkb.Label(parent, text="Select Country:").grid(row=0, column=0, padx=5, pady=5)
        self.country_combo = ClickableCombobox(parent)
        self.country_combo.grid(row=0, column=1, padx=5)
        self.load_country_combo()
        
        button_frame = ttkb.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=3, pady=5)
        
        ttkb.Button(button_frame, text="Add Player", command=self.add_player, bootstyle=INFO).pack(side=tk.LEFT, padx=5)
        ttkb.Button(button_frame, text="Edit Player", command=self.edit_player, bootstyle=WARNING).pack(side=tk.LEFT, padx=5)
        ttkb.Button(button_frame, text="Delete Player", command=self.delete_player, bootstyle=DANGER).pack(side=tk.LEFT, padx=5)
        
        self.players_tree = ttkb.Treeview(parent, 
            columns=('ID', 'Name', 'Role', 'Batting', 'Bowling', 'WK'), 
            show='headings', 
            bootstyle=PRIMARY)
        
        self.players_tree.heading('ID', text='ID')
        self.players_tree.heading('Name', text='Name')
        self.players_tree.heading('Role', text='Role')
        self.players_tree.heading('Batting', text='Batting')
        self.players_tree.heading('Bowling', text='Bowling')
        self.players_tree.heading('WK', text='WK')
        
        self.players_tree.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        self.country_combo.bind('<<ComboboxSelected>>', self.load_players)
    
    def load_country_combo(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT country_name FROM countries")
        self.country_combo['values'] = [row[0] for row in c.fetchall()]
        conn.close()
    
    def add_player(self):
        country = self.country_combo.get()
        if country:
            country_id = self.get_current_country_id()
            PlayerEditor(self.root, country_id)
            self.load_players()
    
    def edit_player(self):
        selected = self.players_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a player to edit")
            return
        
        item = self.players_tree.item(selected[0])
        player_id = item['values'][0]
        PlayerEditor(self.root, self.get_current_country_id(), player_id)
        self.load_players()
    
    def delete_player(self):
        selected = self.players_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a player to delete")
            return
        
        item = self.players_tree.item(selected[0])
        player_id = item['values'][0]
        
        if messagebox.askyesno("Confirm Delete", "Delete this player permanently?"):
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            c.execute("DELETE FROM players WHERE player_id = ?", (player_id,))
            conn.commit()
            conn.close()
            self.load_players()
    
    def get_current_country_id(self):
        country = self.country_combo.get()
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT country_id FROM countries WHERE country_name = ?", (country,))
        country_id = c.fetchone()[0]
        conn.close()
        return country_id
    
    def load_players(self, event=None):
        self.players_tree.delete(*self.players_tree.get_children())
        country = self.country_combo.get()
        if not country:
            return
        
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute('''SELECT player_id, player_name, role, batting_hand, 
                     bowling_arm || " " || bowling_style, 
                     CASE WHEN is_wicketkeeper THEN 'Yes' ELSE 'No' END
                     FROM players 
                     WHERE country_id = (SELECT country_id FROM countries WHERE country_name = ?)''',
                (country,))
        for row in c.fetchall():
            self.players_tree.insert('', 'end', values=row)
        conn.close()
    
    def start_new_innings(self):
        try:
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            
            # Get current innings number
            innings = self.match_data.innings
            
            # Determine batting/bowling teams
            batting_team = self.match_data.team1 if innings % 2 == 1 else self.match_data.team2
            bowling_team = self.match_data.team2 if innings % 2 == 1 else self.match_data.team1
            
            c.execute('''INSERT INTO innings 
                    (match_id, innings, batting_team, bowling_team)
                    VALUES (?, ?, ?, ?)''',
                    (self.match_data.match_id,
                        innings,
                        batting_team,
                        bowling_team))
            
            self.match_data.innings_id = c.lastrowid
            conn.commit()
            
            #print(f"Started innings {innings} (ID: {self.match_data.innings_id})")
            #print(f"Current match_data: {vars(self.match_data)}")
            
            # Reset innings-specific counters
            self.match_data.current_over = 0
            self.match_data.balls_this_over = 0
            self.match_data.wickets = 0
            self.match_data.total_runs = 0
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start innings: {str(e)}")
        finally:
            conn.close()

    def new_match(self):
        MatchSetup(self.root, self)
        
    def view_matches(self):
        MatchViewer(self.root, self)

class ClickableCombobox(ttkb.Combobox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<Button-1>', self._on_click)
    
    def _on_click(self, event):
        if self['state'] == 'readonly':
            self.event_generate('<Down>')

# ================= Database Setup =================
def initialize_database():
    conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
    c = conn.cursor()
    
    # Core Tables
    c.execute('''CREATE TABLE IF NOT EXISTS countries (
                country_id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_name TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                country_id INTEGER,
                role TEXT CHECK(role IN ('Batter', 'Bowler', 'Allrounder')),
                batting_hand TEXT CHECK(batting_hand IN ('Left', 'Right')),
                bowling_arm TEXT CHECK(bowling_arm IN ('Left', 'Right')),
                bowling_style TEXT CHECK(bowling_style IN ('Pace', 'Medium', 'Off Spin', 'Leg Spin')),
                is_wicketkeeper BOOLEAN DEFAULT 0,
                FOREIGN KEY(country_id) REFERENCES countries(country_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS tournaments (
                tournament_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_name TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_a INTEGER,
                team_b INTEGER,
                match_date TEXT,
                venue TEXT,
                tournament_id INTEGER,
                is_training INTEGER DEFAULT 0,
                toss_winner INTEGER,
                toss_decision TEXT,
                result TEXT,
                winner_id INTEGER,
                margin TEXT,
                player_of_match INTEGER,
                rain_interrupted INTEGER DEFAULT 0,
                adjusted_overs INTEGER,
                adjusted_target INTEGER,
                FOREIGN KEY(team_a) REFERENCES countries(country_id),
                FOREIGN KEY(team_b) REFERENCES countries(country_id),
                FOREIGN KEY(tournament_id) REFERENCES tournaments(tournament_id),
                FOREIGN KEY(winner_id) REFERENCES countries(country_id),
                FOREIGN KEY(player_of_match) REFERENCES players(player_id))''')
    
    c.execute("""CREATE TABLE IF NOT EXISTS player_match_roles (
                match_id INTEGER,
                team_id INTEGER,
                player_id INTEGER,
                batting_position INTEGER DEFAULT 0,
                role TEXT,
                is_captain BOOLEAN DEFAULT 0,
                is_keeper BOOLEAN DEFAULT 0,
                PRIMARY KEY (match_id, team_id, player_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (team_id) REFERENCES countries(country_id),
                FOREIGN KEY (match_id) REFERENCES matches(match_id))""")


    
    c.execute('''CREATE TABLE IF NOT EXISTS innings (
                innings_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings INTEGER,
                batting_team INTEGER,
                bowling_team INTEGER,
                total_runs INTEGER DEFAULT 0,
                wickets INTEGER DEFAULT 0,
                overs_bowled REAL DEFAULT 0.0,
                extras INTEGER DEFAULT 0,
                completed INTEGER DEFAULT 0,
                saved_state TEXT,
                FOREIGN KEY(match_id) REFERENCES matches(match_id))''')
    
    # Ball Events with Enhanced Tracking
    c.execute('''CREATE TABLE IF NOT EXISTS ball_events (
                ball_id INTEGER PRIMARY KEY AUTOINCREMENT,
                innings_id INTEGER,
                over_number REAL,
                innings INTEGER,
                balls_this_over INTEGER,
                ball_number INTEGER,
                batter_id INTEGER,
                non_striker_id INTEGER,
                bowler_id INTEGER,
                fielder_id INTEGER,
                runs INTEGER,
                extras TEXT,
                shot_type TEXT,
                footwork TEXT,
                shot_selection TEXT,
                aerial BOOLEAN,
                dismissal_type TEXT,
                dismissed_player_id INTEGER,
                pitch_x REAL,
                pitch_y REAL,
                shot_x REAL,
                shot_y REAL,
                delivery_type TEXT,
                fielding_style TEXT,
                edged BOOLEAN DEFAULT 0,
                ball_missed BOOLEAN DEFAULT 0,
                clean_hit BOOLEAN DEFAULT 0,
                wides INTEGER DEFAULT 0,
                no_balls INTEGER DEFAULT 0,
                free_hit INTEGER DEFAULT 0,
                byes INTEGER DEFAULT 0,
                leg_byes INTEGER DEFAULT 0,
                penalty_runs INTEGER DEFAULT 0,
                dot_balls INTEGER DEFAULT 0,
                expected_runs INTEGER DEFAULT 0,
                expected_wicket REAL DEFAULT 0.00,
                batting_bpi REAL DEFAULT 0.00,
                bowling_bpi REAL DEFAULT 0.00,
                batting_intent_score REAL DEFAULT 0.00,
                batting_position INTEGER DEFAULT 0,
                bowling_order INTEGER DEFAULT 0,
                batter_blind_turn INTEGER DEFAULT 0,
                non_striker_blind_turn INTEGER DEFAULT 0,
                over_the_wicket INTEGER DEFAULT 0,
                around_the_wicket INTEGER DEFAULT 0,
                is_powerplay INTEGER DEFAULT 0,
                is_middle_overs INTEGER DEFAULT 0,
                is_death_overs INTEGER DEFAULT 0,
                FOREIGN KEY(innings_id) REFERENCES innings(innings_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS partnerships (
                partnership_id INTEGER PRIMARY KEY AUTOINCREMENT,
                innings_id INTEGER,
                start_wicket INTEGER,
                batter1_id INTEGER,
                batter2_id INTEGER,
                runs INTEGER DEFAULT 0,
                balls INTEGER DEFAULT 0,
                dots INTEGER DEFAULT 0,
                ones INTEGER DEFAULT 0,
                twos INTEGER DEFAULT 0,
                threes INTEGER DEFAULT 0,
                fours INTEGER DEFAULT 0,
                sixes INTEGER DEFAULT 0,
                start_over REAL DEFAULT 0.0,
                end_over REAL DEFAULT 0.0,
                opponent_team INTEGER,
                unbeaten INTEGER DEFAULT 0,
                FOREIGN KEY(innings_id) REFERENCES innings(innings_id)
            )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS non_ball_dismissals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings_id INTEGER,
                player_id INTEGER,
                dismissal_type TEXT,
                over_number INTEGER
            )''')
    

    # Analysis Tables
    c.execute('''CREATE TABLE IF NOT EXISTS shot_analysis (
                ball_id INTEGER PRIMARY KEY,
                is_beaten BOOLEAN DEFAULT 0,
                shot_risk TEXT CHECK(shot_risk IN ('safe', 'risky', 'reckless')),
                FOREIGN KEY(ball_id) REFERENCES ball_events(ball_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS fielding_contributions (
                contribution_id INTEGER PRIMARY KEY,
                ball_id INTEGER,
                fielder_id INTEGER,
                boundary_saved BOOLEAN DEFAULT 0,
                FOREIGN KEY (ball_id) REFERENCES ball_events(ball_id),
                FOREIGN KEY (fielder_id) REFERENCES players(player_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS player_pressure_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ball_id INTEGER,
                player_id INTEGER,
                team_role TEXT, 
                pressure_type TEXT,
                pressure_value REAL,
                reason TEXT,
                FOREIGN KEY(ball_id) REFERENCES ball_events(ball_id),
                FOREIGN KEY(player_id) REFERENCES players(player_id))''')
        
    # Indexes for Performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_batsman ON ball_events (batter_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_bowler ON ball_events (bowler_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_phase ON ball_events (is_powerplay, is_middle_overs, is_death_overs)''')
    
    # Fielding Events Setup
    c.execute('''CREATE TABLE IF NOT EXISTS fielding_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS ball_fielding_events (
                ball_id INTEGER,
                event_id INTEGER,
                PRIMARY KEY (ball_id, event_id),
                FOREIGN KEY (ball_id) REFERENCES ball_events(ball_id),
                FOREIGN KEY (event_id) REFERENCES fielding_events(event_id))''')
    
   
    try:
       
        # Insert standard fielding events
        fielding_events = [
            'Clean Stop/Pick Up','Catch','Run Out', 'Taken Half Chance','Direct Hit','Drop Catch',
            'Missed Catch','Missed Run Out', 'Missed Half Chance','Fumble','Missed Fielding',
            'Overthrow','Boundary Save', 'Stumping', 'Missed Stumping'
        ]
        c.executemany('''INSERT OR IGNORE INTO fielding_events (event_name)
                    VALUES (?)''', [(e,) for e in fielding_events])
        
        conn.commit()
        

    except sqlite3.Error as e:
        print("Error creating test data:", e)
    
    conn.commit()
    conn.close()

# ================= Match Viewer ==================
class MatchViewer:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("View Matches")
        self.window.geometry("700x550")
        self.setup_ui()

    def setup_ui(self):
        ttkb.Label(self.window, text="Select a Match:", font=("Helvetica", 12, "bold")).pack(pady=10)

        self.match_listbox = tk.Listbox(self.window, width=90, height=20)
        self.match_listbox.pack(padx=10, pady=10)

        self.load_matches()

        button_frame = ttkb.Frame(self.window)
        button_frame.pack(pady=10)

        ttkb.Button(button_frame, text="View Balls", command=self.open_ball_viewer, bootstyle="primary").grid(row=0, column=0, padx=5)
        ttkb.Button(button_frame, text="Resume Match", command=self.resume_match, bootstyle="warning").grid(row=0, column=1, padx=5)

    def load_matches(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Fetch matches with at least one innings and its completion status
        c.execute('''
            SELECT m.match_id, m.match_date, c1.country_name, c2.country_name, 
                   COALESCE(MAX(i.completed), 1) AS is_completed
            FROM matches m
            JOIN countries c1 ON m.team_a = c1.country_id
            JOIN countries c2 ON m.team_b = c2.country_id
            LEFT JOIN innings i ON m.match_id = i.match_id
            GROUP BY m.match_id
            ORDER BY m.match_date DESC
        ''')

        self.matches = []
        for row in c.fetchall():
            match_id, date, team_a, team_b, completed = row
            display = f"{date} — {team_a} vs {team_b} {'✅ Completed' if completed else '🟡 In Progress'}"
            self.matches.append((match_id, completed))
            self.match_listbox.insert(tk.END, display)

        conn.close()

    def open_ball_viewer(self):
        selection = self.match_listbox.curselection()
        if not selection:
            messagebox.showwarning("Select a Match", "Please select a match.")
            return

        match_id, _ = self.matches[selection[0]]
        BallEditor(self.window, self.app, match_id)

    def resume_match(self):
        selection = self.match_listbox.curselection()
        if not selection:
            messagebox.showwarning("Select a Match", "Please select a match.")
            return

        match_id, completed = self.matches[selection[0]]
        if completed:
            messagebox.showinfo("Match Completed", "This match is already completed and cannot be resumed.")
            return

        try:
            #print(f"[DEBUG] Loading match {match_id}")
            match_data = MatchData.load_from_database(match_id)
            #print("[DEBUG] MatchData keys:", list(vars(match_data).keys()))
            self.window.destroy()
            BallByBallInterface(self.parent, self.app, match_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Could not resume match:\n{e}")

class BallEditor:
    def __init__(self, parent, app, match_id):
        self.parent = parent
        self.app = app  # ✅ Store app reference
        self.match_id = match_id
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Match Balls")
        self.window.geometry("800x600")

        self.tree = ttk.Treeview(self.window, columns=("Innings", "Over", "Ball", "Batter", "Runs", "Wicket", "Bowler"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttkb.Button(self.window, text="Edit Selected Ball", command=self.edit_selected_ball, bootstyle="warning").pack(pady=5)

        self.load_balls()

    def load_balls(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        c.execute('''
            SELECT be.ball_id, be.innings_id, i.innings, be.balls_this_over, be.ball_number, p1.player_name, be.runs, be.dismissal_type, p2.player_name
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            LEFT JOIN players p1 ON be.batter_id = p1.player_id
            LEFT JOIN players p2 ON be.bowler_id = p2.player_id
            WHERE i.match_id = ?
            ORDER BY be.innings_id, be.balls_this_over, be.ball_number
        ''', (self.match_id,))

        self.ball_data = []
        for row in c.fetchall():
            ball_id, innings_id, innings_num, over, ball_num, batter, runs, dismissal, bowler = row
            self.ball_data.append((ball_id, innings_id))  # Track ball_id for editing
            self.tree.insert("", tk.END, values=(innings_num, over, ball_num, batter, runs, dismissal or "", bowler))

        conn.close()

    def edit_selected_ball(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a ball to edit.")
            return

        idx = self.tree.index(selection[0])
        ball_id, innings_id = self.ball_data[idx]
        BallEditForm(self.parent, self.app, ball_id)

class BallEditForm:
    def __init__(self, parent, app, ball_id):
        self.parent = parent
        self.app = app  # Store reference to main app
        self.ball_id = ball_id
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Ball Details")
        self.window.geometry("500x500")

        self.fields = {}

        self.load_ball_data()

    def load_ball_data(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        c.execute('''
            SELECT 
                be.batter_id, be.bowler_id, be.runs, be.extras, 
                be.dismissal_type, be.balls_this_over, be.ball_number,
                be.wides, be.no_balls, be.byes, be.leg_byes
            FROM ball_events be
            WHERE be.ball_id = ?
        ''', (self.ball_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            messagebox.showerror("Error", "Ball data not found.")
            self.window.destroy()
            return

        (self.batter_id, self.bowler_id, runs, extras, dismissal_type, 
         over, number, wides, no_balls, byes, leg_byes) = row

        # Draw the form
        self.draw_form({
            "Runs": runs,
            "Extras": extras,
            "Wides": wides,
            "No Balls": no_balls,
            "Byes": byes,
            "Leg Byes": leg_byes,
            "Dismissal": dismissal_type or "",
            "Ball Over": over,
            "Ball Number": number
        })

    def draw_form(self, data):
        form_frame = tk.Frame(self.window)
        form_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        for idx, (label, value) in enumerate(data.items()):
            tk.Label(form_frame, text=label).grid(row=idx, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(form_frame)
            entry.grid(row=idx, column=1, pady=5)
            entry.insert(0, str(value))
            self.fields[label] = entry

        ttkb.Button(self.window, text="Save Changes", command=self.save_changes, bootstyle="success").pack(pady=10)

    def save_changes(self):
        try:
            runs = int(self.fields["Runs"].get())
            extras = int(self.fields["Extras"].get())
            wides = int(self.fields["Wides"].get())
            no_balls = int(self.fields["No Balls"].get())
            byes = int(self.fields["Byes"].get())
            leg_byes = int(self.fields["Leg Byes"].get())
            dismissal_type = self.fields["Dismissal"].get().strip()
        except ValueError:
            messagebox.showerror("Error", "Numeric fields must be valid integers.")
            return

        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Update ball_events
        c.execute('''
            UPDATE ball_events
            SET runs = ?, extras = ?, wides = ?, no_balls = ?, byes = ?, leg_byes = ?, dismissal_type = ?
            WHERE ball_id = ?
        ''', (runs, extras, wides, no_balls, byes, leg_byes, dismissal_type or None, self.ball_id))

        # Optionally update fielding events here if you added a "Fielding Events" field

        conn.commit()
        conn.close()

        messagebox.showinfo("Saved", "Ball updated successfully.")

        # ✅ Prompt to trigger full cascade AFTER success
        choice = messagebox.askyesno("Cascade Recalculation", "Recalculate all balls after this one in this innings?")
        if choice:
            self.recalculate_from_ball(self.ball_id)

        # ✅ Finally, close the edit window
        self.window.destroy()

    def recalculate_from_ball(self, start_ball_id):
        self.app.calculate_bpi(ball_events, current_ball)
        self.app.save_individual_pressure_impact(ball_events, current_ball)


        #print("📦 Recalculating all subsequent balls...")

        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Step 1: Find innings_id and ball position
        c.execute('''
            SELECT innings_id, balls_this_over, ball_number
            FROM ball_events
            WHERE ball_id = ?
        ''', (start_ball_id,))
        start_row = c.fetchone()
        if not start_row:
            #("❌ Could not find start ball info.")
            conn.close()
            return

        innings_id, start_over, start_number = start_row

        # Step 2: Load all balls in innings, ordered
        c.execute('''
            SELECT * FROM ball_events
            WHERE innings_id = ?
            ORDER BY ball_over ASC, ball_number ASC
        ''', (innings_id,))
        all_balls = c.fetchall()
        columns = [desc[0] for desc in c.description]

        # Step 3: Convert to dicts, find start index
        ball_dicts = [dict(zip(columns, row)) for row in all_balls]
        start_index = next(i for i, b in enumerate(ball_dicts) if b["ball_id"] == start_ball_id)

        # Step 4: Loop through each subsequent ball and recalculate
        ball_events = ball_dicts[:start_index]  # context history
        for current_ball in ball_dicts[start_index:]:
            ball_id = current_ball["ball_id"]

            # Update: fielding events
            c.execute("SELECT event_type FROM ball_fielding_events WHERE ball_id = ?", (ball_id,))
            current_ball["fielding_events"] = [r[0] for r in c.fetchall()]

            # Delete old pressure data
            c.execute("DELETE FROM player_pressure_impact WHERE ball_id = ?", (ball_id,))

            # Recalculate and re-save
            self.app.calculate_bpi(ball_events, current_ball)  # Updates DB directly
            self.app.save_individual_pressure_impact(ball_events, current_ball)

            # Append to event history
            ball_events.append(current_ball)

        conn.commit()
        conn.close()

        #print("✅ Cascade complete.")
        messagebox.showinfo("Recalculation Complete", "All subsequent balls have been recalculated successfully.")

# ================= Player Editor =================
class PlayerEditor:
    def __init__(self, parent, country_id, player_id=None):
        self.window = ttkb.Toplevel(parent)
        self.window.title("Edit Player" if player_id else "Add Player")
        self.country_id = country_id
        self.player_id = player_id
        
        # Form elements
        ttkb.Label(self.window, text="Player Name:").grid(row=0, column=0, padx=5, pady=5)
        self.name_entry = ttkb.Entry(self.window)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttkb.Label(self.window, text="Role:").grid(row=1, column=0, padx=5, pady=5)
        self.role_combo = ClickableCombobox(self.window, values=["Batter", "Bowler", "Allrounder"])
        self.role_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttkb.Label(self.window, text="Batting Hand:").grid(row=2, column=0, padx=5, pady=5)
        self.batting_combo = ClickableCombobox(self.window, values=["Left", "Right"])
        self.batting_combo.grid(row=2, column=1, padx=5, pady=5)
        
        ttkb.Label(self.window, text="Bowling Arm:").grid(row=3, column=0, padx=5, pady=5)
        self.bowling_arm_combo = ClickableCombobox(self.window, values=["Left", "Right"])
        self.bowling_arm_combo.grid(row=3, column=1, padx=5, pady=5)
        
        ttkb.Label(self.window, text="Bowling Style:").grid(row=4, column=0, padx=5, pady=5)
        self.style_combo = ClickableCombobox(self.window, values=["Pace", "Medium", "Off Spin", "Leg Spin"])
        self.style_combo.grid(row=4, column=1, padx=5, pady=5)
        
        self.wk_var = tk.BooleanVar()
        ttkb.Checkbutton(self.window, text="Wicket Keeper", variable=self.wk_var).grid(row=5, column=1, padx=5, pady=5)
        
        ttkb.Button(self.window, text="Save", command=self.save, bootstyle=SUCCESS).grid(row=6, column=0, columnspan=2, pady=10)
        
        if player_id:
            self.load_existing_data()
    
    def load_existing_data(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute('''SELECT player_name, role, batting_hand, bowling_arm, bowling_style, is_wicketkeeper 
                     FROM players WHERE player_id = ?''', (self.player_id,))
        data = c.fetchone()
        conn.close()
        
        if data:
            self.name_entry.insert(0, data[0])
            self.role_combo.set(data[1])
            self.batting_combo.set(data[2])
            self.bowling_arm_combo.set(data[3])
            self.style_combo.set(data[4])
            self.wk_var.set(data[5])
    
    def save(self):
        data = (
            self.name_entry.get(),
            self.role_combo.get(),
            self.batting_combo.get(),
            self.bowling_arm_combo.get(),
            self.style_combo.get(),
            int(self.wk_var.get())
        )
        
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        try:
            if self.player_id:
                c.execute('''UPDATE players SET 
                            player_name = ?,
                            role = ?,
                            batting_hand = ?,
                            bowling_arm = ?,
                            bowling_style = ?,
                            is_wicketkeeper = ?
                            WHERE player_id = ?''', (*data, self.player_id))
            else:
                c.execute('''INSERT INTO players 
                            (player_name, role, batting_hand, bowling_arm, bowling_style, is_wicketkeeper, country_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''', (*data, self.country_id))
            conn.commit()
            messagebox.showinfo("Success", "Player saved successfully")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save player: {str(e)}")
        finally:
            conn.close()
       
# ================= Match Setup =================
class MatchSetup:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.window = ttkb.Toplevel(parent)
        self.window.title("New Match Setup")
        self.window.grab_set()
        
        self.create_widgets()
        self.load_data()
        self.window.minsize(400, 400)

    def create_widgets(self):
        main_frame = ttkb.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Team Selection
        ttkb.Label(main_frame, text="Team 1:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.team1_combo = ClickableCombobox(main_frame)
        self.team1_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttkb.Label(main_frame, text="Team 2:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.team2_combo = ClickableCombobox(main_frame)
        self.team2_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        # Date Entry
        ttkb.Label(main_frame, text="Match Date:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.date_entry = ttkb.DateEntry(main_frame, dateformat="%Y-%m-%d")
        self.date_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        #self.date_entry.entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        # Tournament Selection
        ttkb.Label(main_frame, text="Tournament:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.tournament_combo = ClickableCombobox(main_frame)
        self.tournament_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

        # New Tournament Entry
        self.new_tournament_var = tk.BooleanVar()
        self.new_tournament_check = ttkb.Checkbutton(
            main_frame, 
            text="New Tournament", 
            variable=self.new_tournament_var,
            command=self.toggle_tournament
        )
        self.new_tournament_check.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        self.new_tournament_entry = ttkb.Entry(main_frame)
        
        # Overs per Innings
        ttkb.Label(main_frame, text="Overs per Innings:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.overs_entry = ttkb.Spinbox(main_frame, from_=1, to=50, width=5)
        self.overs_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        self.overs_entry.set(20)

        # Training Match Toggle
        self.is_training_var = tk.BooleanVar()
        self.training_check = ttkb.Checkbutton(
            main_frame,
            text="Training Match",
            variable=self.is_training_var
        )
        self.training_check.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)


        # Continue Button
        ttkb.Button(main_frame, 
                   text="Continue to Team Selection", 
                   command=self.validate_and_proceed, 
                   bootstyle=SUCCESS).grid(row=7, column=0, columnspan=2, pady=15, sticky=tk.EW)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)

    def load_data(self):
        """Load teams and tournaments from database"""
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        try:
            # Load teams
            c = conn.cursor()
            c.execute("SELECT country_name FROM countries")
            teams = [row[0] for row in c.fetchall()]
            self.team1_combo['values'] = teams
            self.team2_combo['values'] = teams
            
            # Load tournaments
            c.execute("SELECT tournament_name FROM tournaments")
            tournaments = [row[0] for row in c.fetchall()]
            self.tournament_combo['values'] = tournaments
            
            # Set default selections
            if teams:
                self.team1_combo.current(0)
                if len(teams) > 1:
                    self.team2_combo.current(1)
            if tournaments:
                self.tournament_combo.current(0)
                
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to load data: {str(e)}")
        finally:
            conn.close()

    def toggle_tournament(self):
        """Switch between existing and new tournament entry"""
        if self.new_tournament_var.get():
            self.tournament_combo.grid_remove()
            self.new_tournament_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        else:
            self.new_tournament_entry.grid_remove()
            self.tournament_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

    def validate_and_proceed(self):
        """Validate inputs and proceed to team selection"""
        team1 = self.team1_combo.get()
        team2 = self.team2_combo.get()
        date = self.date_entry.entry.get()
        tournament = self.tournament_combo.get()
        is_training = self.is_training_var.get()
        new_tournament = self.new_tournament_entry.get()
        overs = int(self.overs_entry.get())


        # Validate teams
        errors = []
        if not team1 or not team2:
            errors.append("Please select both teams")
        elif team1 == team2:
            errors.append("Teams must be different")

        # Validate tournament
        if self.new_tournament_var.get():
            if not new_tournament:
                errors.append("Please enter new tournament name")
            else:
                tournament = new_tournament
        elif not tournament:
            errors.append("Please select or create a tournament")

        # Validate overs
        try:
            overs = int(overs)
            if not 1 <= overs <= 50:
                errors.append("Overs must be between 1-50")
        except ValueError:
            errors.append("Invalid overs value")

        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return

        # Save new tournament if needed
        if self.new_tournament_var.get():
            try:
                conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
                c = conn.cursor()
                c.execute("INSERT INTO tournaments (tournament_name) VALUES (?)", (tournament,))
                conn.commit()
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Tournament already exists!")
                return
            finally:
                conn.close()

        # Get country IDs
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT country_id FROM countries WHERE country_name = ?", (team1,))
        team1_id = c.fetchone()[0]
        c.execute("SELECT country_id FROM countries WHERE country_name = ?", (team2,))
        team2_id = c.fetchone()[0]
        



        # Insert match and get match_id
        c.execute('''INSERT INTO matches (team_a, team_b, match_date, tournament_id, is_training)
                    VALUES (?, ?, ?, (SELECT tournament_id FROM tournaments WHERE tournament_name = ?),?)''',
                (team1_id, team2_id, date, tournament, is_training))
        match_id = c.lastrowid  # This is critical
        conn.commit()

        # Initialize MatchData with match_id
        match_info = {
            'match_id': match_id,  # Key added
            'team1': team1,
            'team2':team2,
            'team1_id': team1_id,
            'team2_id': team2_id,
            'date': date,
            'tournament':c.execute("SELECT tournament_id FROM tournaments WHERE tournament_name = ?", 
                                     (tournament,)).fetchone()[0],
            'overs_per_innings': overs
        }

        match_info['overs_phases'] = define_game_phases(overs)
        match_info['is_training'] = bool(is_training)



        self.app.match_data = MatchData(match_info)  # Pass match_info
        # Debug before starting innings
        #print(f"[DEBUG] Pre-innings MatchData: {vars(self.app.match_data)}")
        
        self.app.start_new_innings()

        # After starting innings
        #print(f"[DEBUG] Post-innings MatchData: {vars(self.app.match_data)}")
        #print(f"[DEBUG] Innings ID in MatchSetup: {self.app.match_data.innings_id}")    

        conn.close()

        

        self.window.destroy()
        TeamSelector(self.parent, self.app, self.app.match_data)

class TeamSelector:
    def __init__(self, parent, app, match_data):
        self.parent = parent
        self.app = app
        self.match_data = match_data
        self.window = ttkb.Toplevel(parent)
        self.window.title("Select Playing XI")
        self.window.geometry("1200x600")
        self.window.grab_set()

        self.team1_playing = []
        self.team1_twelfth = []
        self.team2_playing = []
        self.team2_twelfth = []
        self.selection_vars = {}  # {player_id: IntVar()}
        self.team1_captain_var = tk.IntVar()
        self.team2_captain_var = tk.IntVar()
        self.team1_keeper_var = tk.IntVar()
        self.team2_keeper_var = tk.IntVar()

        self.create_widgets()
        self.load_players()

    def create_widgets(self):
        main_frame = ttkb.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.team1_frame = ttkb.LabelFrame(main_frame, text=self.match_data.team1, padding=10)
        self.team1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.team2_frame = ttkb.LabelFrame(main_frame, text=self.match_data.team2, padding=10)
        self.team2_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttkb.Button(self.window, text="Confirm Selection", 
                   command=self.validate_selection, 
                   bootstyle=SUCCESS,
                   width=20).pack(pady=10)

    def load_players(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        try:
            c = conn.cursor()
            c.execute('''SELECT player_id, player_name FROM players WHERE country_id = ?''', (self.match_data.team1_id,))
            team1_players = c.fetchall()
            self.create_team_selection(self.team1_frame, team1_players, is_team1=True)

            c.execute('''SELECT player_id, player_name FROM players WHERE country_id = ?''', (self.match_data.team2_id,))
            team2_players = c.fetchall()
            self.create_team_selection(self.team2_frame, team2_players, is_team1=False)
        finally:
            conn.close()

    def create_team_selection(self, parent, players, is_team1):
        container = ttkb.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttkb.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttkb.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for player_id, name in players:
            frame = ttkb.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)

            var = tk.IntVar(value=0)
            self.selection_vars[player_id] = var

            ttkb.Label(frame, text=name, width=25).pack(side=tk.LEFT)
            ttkb.Radiobutton(frame, text="Playing", variable=var, value=1, bootstyle="success-outline-toolbutton", width=10).pack(side=tk.LEFT, padx=2)
            ttkb.Radiobutton(frame, text="12th", variable=var, value=2, bootstyle="warning-outline-toolbutton", width=6).pack(side=tk.LEFT, padx=2)
            ttkb.Radiobutton(frame, text="❌", variable=var, value=0, bootstyle="danger-outline-toolbutton", width=3).pack(side=tk.RIGHT, padx=2)

            captain_var = self.team1_captain_var if is_team1 else self.team2_captain_var
            keeper_var = self.team1_keeper_var if is_team1 else self.team2_keeper_var

            ttkb.Radiobutton(frame, text="Captain", variable=captain_var, value=player_id, bootstyle="info-outline-toolbutton", width=8).pack(side=tk.LEFT, padx=2)
            ttkb.Radiobutton(frame, text="Keeper", variable=keeper_var, value=player_id, bootstyle="secondary-outline-toolbutton", width=8).pack(side=tk.LEFT, padx=2)

    def validate_selection(self):
        is_training = getattr(self.match_data, 'is_training', False)

        self.team1_playing = []
        self.team1_twelfth = []
        self.team2_playing = []
        self.team2_twelfth = []

        # Shared DB connection
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db", timeout=10)
        c = conn.cursor()

        for player_id, var in self.selection_vars.items():
            if player_id <= 0:
                continue  # skip invalid IDs

            c.execute("SELECT country_id FROM players WHERE player_id = ?", (player_id,))
            result = c.fetchone()
            if not result:
                continue  # skip non-existent players

            team_id = result[0]
            if var.get() == 1:  # Playing
                if team_id == self.match_data.team1_id:
                    self.team1_playing.append(player_id)
                else:
                    self.team2_playing.append(player_id)
            elif var.get() == 2:  # 12th
                if team_id == self.match_data.team1_id:
                    self.team1_twelfth.append(player_id)
                else:
                    self.team2_twelfth.append(player_id)

        # ❌ Strict rules in normal mode
        errors = []
        if not is_training:
            if len(self.team1_playing) != 11:
                errors.append(f"{self.match_data.team1} must have exactly 11 playing players")
            if len(self.team2_playing) != 11:
                errors.append(f"{self.match_data.team2} must have exactly 11 playing players")
            if len(self.team1_twelfth) > 1:
                errors.append(f"{self.match_data.team1} can only have one 12th player")
            if len(self.team2_twelfth) > 1:
                errors.append(f"{self.match_data.team2} can only have one 12th player")

        if errors:
            conn.close()
            messagebox.showerror("Selection Error", "\n".join(errors))
            return

        self.match_data.selected_players = {
            self.match_data.team1: self.team1_playing,
            self.match_data.team2: self.team2_playing
        }
        self.match_data.team1_twelfth = self.team1_twelfth
        self.match_data.team2_twelfth = self.team2_twelfth

        # Save player roles
        for pos, player_id in enumerate(self.team1_playing):
            if player_id <= 0:
                continue
            is_captain = 1 if player_id == self.team1_captain_var.get() else 0
            is_keeper = 1 if player_id == self.team1_keeper_var.get() else 0
            c.execute("""
                INSERT OR REPLACE INTO player_match_roles 
                (match_id, team_id, player_id, batting_position, role, is_captain, is_keeper)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (self.match_data.match_id, self.match_data.team1_id, player_id, pos + 1, 'Batter', is_captain, is_keeper))

        for pos, player_id in enumerate(self.team2_playing):
            if player_id <= 0:
                continue
            is_captain = 1 if player_id == self.team2_captain_var.get() else 0
            is_keeper = 1 if player_id == self.team2_keeper_var.get() else 0
            c.execute("""
                INSERT OR REPLACE INTO player_match_roles 
                (match_id, team_id, player_id, batting_position, role, is_captain, is_keeper)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (self.match_data.match_id, self.match_data.team2_id, player_id, pos + 1, 'Batter', is_captain, is_keeper))

        conn.commit()
        conn.close()

        self.window.destroy()
        TossSetup(self.parent, self.app, self.match_data)

class TossSetup:
    def __init__(self, parent, app, match_data):
        self.parent = parent
        self.app = app
        self.match_data = match_data  # Using MatchData class
        self.window = ttkb.Toplevel(parent)
        self.window.title("Toss Setup")
        self.window.grab_set()
        #print(f"[DEBUG] Innings ID at toss setup: {self.app.match_data.innings_id}")
        self.create_widgets()  # This will now work
        
    def create_widgets(self):
        main_frame = ttkb.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttkb.Label(main_frame, text="Toss Winner:").pack(pady=5)
        self.toss_winner = ClickableCombobox(
            main_frame, 
            values=[self.match_data.team1, self.match_data.team2],
            state='readonly'
        )
        self.toss_winner.pack(pady=5)
        
        ttkb.Label(main_frame, text="Toss Decision:").pack(pady=5)
        self.toss_decision = ClickableCombobox(
            main_frame, 
            values=['Bat', 'Field'], 
            state='readonly'
        )
        self.toss_decision.pack(pady=5)
        
        ttkb.Button(main_frame, text="Continue", 
                   command=self.save_toss, 
                   bootstyle=SUCCESS).pack(pady=20)

    def save_toss(self):
        if not all([self.toss_winner.get(), self.toss_decision.get()]):
            messagebox.showerror("Error", "Please complete toss details")
            return

        # Store toss results in match_data
        self.match_data.toss_winner = self.toss_winner.get()
        self.match_data.toss_decision = self.toss_decision.get()

        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        # Insert match and get match_id
        
        c.execute('''UPDATE matches SET 
                   toss_winner = ?,
                   toss_decision = ?
                   WHERE match_id = ?''',
                (self.match_data.toss_winner,
                 self.match_data.toss_decision,
                 self.app.match_data.match_id))
        
        conn.commit()
        
        
        # Determine batting and bowling teams
        if self.toss_decision.get() == 'Bat':
            self.match_data.batting_team = self.match_data.toss_winner
            self.match_data.bowling_team = (
                self.match_data.team2 if self.match_data.toss_winner == self.match_data.team1 
                else self.match_data.team1
            )
        else:
            self.match_data.bowling_team = self.match_data.toss_winner
            self.match_data.batting_team = (
                self.match_data.team2 if self.match_data.toss_winner == self.match_data.team1 
                else self.match_data.team1
            )

        conn.close()
        
        self.window.destroy()
        self.window.after(200, lambda: self.start_ball_by_ball())  # ✅ Delay execution slightly


    def start_ball_by_ball(self):
        OpeningPlayers(self.parent, self.app, self.match_data)
        
# ================= Match Data Class =================
class MatchData:
    def __init__(self, match_data):
        # Existing attributes
        self.match_id = match_data.get("match_id")  # Key fix
        self.innings_id = match_data.get("innings_id")
        self.team1 = match_data['team1']
        self.team2 = match_data['team2']
        self.team1_id = match_data['team1_id']
        self.team2_id = match_data['team2_id']
        self.date = match_data['date']
        self.tournament = match_data['tournament']
        self.is_training = match_data.get('is_training', False)

        self.overs_per_innings = match_data.get('overs_per_innings', 20)
        self.adjusted_overs = match_data.get('adjusted_overs')
        effective_overs = self.adjusted_overs or self.overs_per_innings
        self.overs_phases = define_game_phases(effective_overs)   

        # Correctly initialize selected_players with the playing XIs
        self.selected_players = {
            match_data['team1']: match_data.get('team1_playing', []),
            match_data['team2']: match_data.get('team2_playing', [])
        }
        # New attributes for game situation
        self.current_over = 0.0
        self.balls_this_over = 0
        self.total_balls = 0
        self.total_runs = 0
        self.wickets = 0
        self.striker = None
        self.non_striker = None
        self.current_bowler = None
        self.ball_history = []
        self.batters = {}
        self.bowlers = {}
        self.batting_team = None
        self.bowling_team = None
        self.current_bowler = None
        self.toss_winner = None
        self.toss_decision = None
        self.current_over_balls = []  # Track balls in current over
        self.innings = 1
        self.innings_ended = False 
        self.target_runs = None      # For second innings
        self.required_rr = tk.StringVar(value="0.00")
        self.total_overs = match_data['overs_per_innings']
        self.dismissed_players = set()
        self.retired_not_out_players = set()
        self.last_dismissal = None
        self.bowling_history = []  # Tracks bowling order
        self.team1_twelfth = match_data.get('team1_twelfth', [])
        self.team2_twelfth = match_data.get('team2_twelfth', [])
        self.bowlers = match_data.get('bowlers', {})
        self.bowlers = defaultdict(lambda: {
            'balls': 0, 'runs': 0, 'wickets': 0,
            'dot_balls': 0,
            'wides': 0, 'no_balls': 0
        })
        self.new_batter = None
        self.dismissed_batter = None
        self.current_partnership = {
            'batter1': None,
            'batter2': None,
            'runs': 0,
            'balls': 0,
            'dots': 0,
            'ones': 0,
            'twos': 0,
            'threes': 0,
            'fours': 0,
            'sixes': 0,
            'start_wicket': 1,
            'start_over': 0.0,
            'unbeaten': 0   
        }
        self.partnerships = []
        self.bowler_usage = {}
        self.waiting_for_new_innings_setup = False
        self.is_super_over = False
        self.super_over_round = 0
        self.super_over_stage = None  # "TeamA" or "TeamB"
        self.super_over_scores = []   # list of tuples: [(TeamA_runs, TeamB_runs), ...]
        self.was_rain_delayed = False
        self.adjusted_target = match_data.get('adjusted_target')
        self.over_var = tk.IntVar(value=1)
        

        self.batting_bpi = 0.0
        self.bowling_bpi = 0.0
        
    @staticmethod
    def load_from_database(match_id):
        import sqlite3
        import json
        from collections import defaultdict

        #print(f"[DEBUG] Loading match {match_id}")
        conn = sqlite3.connect("C:/Users/Danielle/Desktop/Cricket Analysis Program/cricket_analysis.db")
        c = conn.cursor()

        # Step 1: Get uncompleted innings
        c.execute('''
            SELECT innings_id, innings, batting_team, bowling_team, saved_state
            FROM innings
            WHERE match_id = ? AND completed = 0
            ORDER BY innings DESC
            LIMIT 1
        ''', (match_id,))
        innings_row = c.fetchone()
        if not innings_row:
            raise ValueError("No incomplete innings found for this match.")

        innings_id, innings_num, batting_team_name, bowling_team_name, saved_state_json = innings_row

        # Step 2: Get match metadata
        c.execute("SELECT team_a, team_b, tournament_id, match_date, adjusted_overs, adjusted_target FROM matches WHERE match_id = ?", (match_id,))
        match_row = c.fetchone()
        if not match_row:
            raise ValueError("Match not found.")

        team1_id, team2_id, tournament_id, match_date, adjusted_overs, adjusted_target = match_row

        # Step 3: Get country names
        c.execute("SELECT country_name FROM countries WHERE country_id = ?", (team1_id,))
        team1_name = c.fetchone()[0]
        c.execute("SELECT country_name FROM countries WHERE country_id = ?", (team2_id,))
        team2_name = c.fetchone()[0]

        #print(f"[DEBUG] From match: team1_id={team1_id}, team2_id={team2_id}")
        #print(f"[DEBUG] Resolved names: team1_name={team1_name}, team2_name={team2_name}")
        #print(f"[DEBUG] From innings: batting={batting_team_name}, bowling={bowling_team_name}")

        # Step 4: Prepare base match_data dict
        match_data_dict = {
            'match_id': match_id,
            'innings': innings_num,
            'innings_id': innings_id,
            'batting_team': batting_team_name,
            'bowling_team': bowling_team_name,
            'team1': team1_name,
            'team2': team2_name,
            'team1_id': team1_id,
            'team2_id': team2_id,
            'tournament': tournament_id,
            'date': match_date,
            'overs_per_innings': adjusted_overs or 20,
            'adjusted_overs': adjusted_overs,
            'adjusted_target': adjusted_target
        }

        match_data = MatchData(match_data_dict)

        # Step 5: Restore state
        if saved_state_json:
            state = json.loads(saved_state_json)

            match_data.total_runs = state.get("total_runs", 0)
            match_data.wickets = state.get("wickets", 0)
            match_data.current_over = state.get("current_over", 0)
            match_data.balls_this_over = state.get("balls_this_over", 0)
            match_data.striker = int(state.get("striker")) if state.get("striker") else None
            match_data.non_striker = int(state.get("non_striker")) if state.get("non_striker") else None
            match_data.current_bowler = int(state.get("current_bowler")) if state.get("current_bowler") else None

            match_data.batters = {int(k): v for k, v in state.get("batters", {}).items()}
            match_data.bowlers = defaultdict(
                lambda: {'balls': 0, 'runs': 0, 'wickets': 0, 'maidens': 0, 'dot_balls': 0, 'wides': 0, 'no_balls': 0},
                {int(k): v for k, v in state.get("bowlers", {}).items()}
            )

            # Partnership fix
            match_data.current_partnership = state.get("current_partnership", None)
            if match_data.current_partnership:
                match_data.current_partnership['batter1'] = int(match_data.current_partnership['batter1']) if match_data.current_partnership['batter1'] else None
                match_data.current_partnership['batter2'] = int(match_data.current_partnership['batter2']) if match_data.current_partnership['batter2'] else None

            match_data.dismissed_players = set(int(pid) for pid in state.get("dismissed_players", []))
            match_data.retired_not_out_players = set(int(pid) for pid in state.get("retired_not_out_players", []))
            match_data.batting_bpi = state.get("batting_bpi", 0.0)
            match_data.bowling_bpi = state.get("bowling_bpi", 0.0)
            match_data.current_over_balls = state.get("current_over_balls", [])
            match_data.waiting_for_new_innings_setup = state.get("waiting_for_new_innings_setup", False)

            # 🧠 Safe fallback for team keys
            selected = state.get("selected_players", {})
            match_data.selected_players = {
                team1_name: [int(pid) for pid in selected.get(team1_name, [])],
                team2_name: [int(pid) for pid in selected.get(team2_name, [])]
            }

            match_data.team1_twelfth = state.get("team1_twelfth", [])
            match_data.team2_twelfth = state.get("team2_twelfth", [])

            # ✅ NEVER overwrite team names with None
            if state.get("batting_team"):
                match_data.batting_team = state.get("batting_team")
            if state.get("bowling_team"):
                match_data.bowling_team = state.get("bowling_team")

        conn.close()

        #print(f"[DEBUG] MatchData keys: {list(match_data.__dict__.keys())}")
        #print(f"[DEBUG] Overs Phases: {match_data.overs_phases}")
        #print(f"[DEBUG] Current Over Float: {match_data.current_over}")
        return match_data   
    
# ================= Opening Players Selection =================
class OpeningPlayers:
    def __init__(self, parent, app, match_data):
        self.parent = parent
        self.app = app
        self.match_data = match_data
        self.window = ttkb.Toplevel(parent)
        self.window.title("Select Opening Players")
        self.window.geometry("400x400")
        self.window.grab_set()
        
               # ❌ Prevent the user from closing the window without selecting players
        self.window.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning(
            "Required", "You must select both opening batters to proceed."))
        # Add debug info
        #print(f"[MATCHDATA INIT] Received: {match_data}")
        #print(f"[OPENING PLAYERS] Received innings_id: {self.match_data.innings_id}")  # Correct access
        self.create_widgets()
        self.load_players()

    def create_widgets(self):
        main_frame = ttkb.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Batters from batting team
        ttkb.Label(main_frame, text="Striker:").grid(row=0, column=0, padx=5, pady=5)
        self.striker_combo = ttkb.Combobox(main_frame, state="readonly")
        self.striker_combo.grid(row=0, column=1, padx=5, pady=5)

        ttkb.Label(main_frame, text="Non-Striker:").grid(row=1, column=0, padx=5, pady=5)
        self.non_striker_combo = ttkb.Combobox(main_frame, state="readonly")
        self.non_striker_combo.grid(row=1, column=1, padx=5, pady=5)

        ttkb.Button(main_frame, text="Start Innings", 
                   command=self.validate_and_start,
                   bootstyle=SUCCESS).grid(row=3, column=0, columnspan=2, pady=10)

    def load_players(self):
        """Load available players excluding 12th man"""
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Load batters (batting team excluding 12th man)
        placeholders = ','.join(['?'] * len(self.match_data.team1_twelfth))
        batter_query = f'''
            SELECT player_id, player_name 
            FROM players 
            WHERE country_id = (
                SELECT country_id 
                FROM countries 
                WHERE country_name = ?
            )
            AND player_id NOT IN ({placeholders})
        '''
        batter_params = (self.match_data.batting_team,) + tuple(self.match_data.team1_twelfth)
        c.execute(batter_query, batter_params)
        batters = c.fetchall()
        
        self.striker_combo['values'] = [name for _, name in batters]
        self.non_striker_combo['values'] = [name for _, name in batters]
        
        conn.close()

    def validate_and_start(self):
        striker_name = self.striker_combo.get()
        non_striker_name = self.non_striker_combo.get()

        if not all([striker_name, non_striker_name]):
            messagebox.showerror("Error", "Please select all required players")
            return

        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Get player IDs as integers
        c.execute("SELECT player_id FROM players WHERE player_name = ?", (striker_name,))
        self.match_data.striker = int(c.fetchone()[0])
        c.execute("SELECT player_id FROM players WHERE player_name = ?", (non_striker_name,))
        self.match_data.non_striker = int(c.fetchone()[0])


        # Initialize batter stats
        self.match_data.batters = {
            self.match_data.striker: {'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'},
            self.match_data.non_striker: {'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'}
        }

        
        # Initialize first partnership with IDs
        self.match_data.current_partnership = {
            'batter1': self.match_data.striker,
            'batter2': self.match_data.non_striker,
            'runs': 0,
            'balls': 0,
            'dots': 0,
            'ones': 0,
            'twos': 0,
            'threes': 0,
            'fours': 0,
            'sixes': 0,
            'start_wicket': 1,
            'start_over': 0.0,
            'unbeaten': 0
        }


        conn.close()
        self.window.destroy()
        BallByBallInterface(self.parent, self.app, self.match_data)

# ================= Ball by Ball Interface =================

class BallByBallInterface:
    def __init__(self, parent, app, match_data):
        self.parent = parent
        self.app = app
        self.match_data = match_data
        self.style = Style(theme='cyborg')  # Initialize style first
        self.window = self.style.master  # Access the style's root window
        self.window = tk.Toplevel(self.window)  # Create child window
        self.window.title("Ball-by-Ball Input")
        self.window.state('zoomed')
        self.history = []
        self.style = ttkb.Style()
        self.style.configure('Treeview', rowheight=25, font=('Helvetica', 10))
        self.style.configure('Treeview.Heading', font=('Helvetica', 11, 'bold'))
        self.style.map('Treeview', background=[('selected', '#347083')])
        
        # Initialize coordinate storage
        self.current_pitch_location = None
        self.current_shot_location = None

        # In save_ball_event():
        #print(f"[DEBUG] Current MatchData ID: {id(self.match_data)}")
        #print(f"[DEBUG] Innings ID at save: {self.match_data.innings_id}")
        self.expected_manually_changed = False

        # Main container using grid exclusively
        main_container = tk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True)  # Pack at root level

        # Configure grid layout
        main_container.grid_columnconfigure(0, weight=0)  # Left panel (fixed)
        main_container.grid_columnconfigure(1, weight=1)  # Center (expanding)
        main_container.grid_columnconfigure(2, weight=0)  # Right panel (fixed)
        main_container.grid_rowconfigure(0, weight=1)

        # Create panels
        self.create_left_graphics_panel(main_container)
        self.create_center_input_panel(main_container)
        self.create_right_game_panel(main_container)

        # Initialize fielding_events dictionary
        self.fielding_events = {}

        # Add this check instead:
        if not hasattr(match_data, 'current_bowler') or not match_data.current_bowler:
            self.select_opening_bowler()
        else:
            self.update_display()

        self.match_data.current_over_balls = []  # Add this line

        # Initialize batters if not set
        if not hasattr(self.match_data, 'batters'):
            self.match_data.batters = {}

        # Initialize bowler stats if missing
        if self.match_data.current_bowler not in self.match_data.bowlers:
            self.match_data.bowlers[self.match_data.current_bowler] = {
                'balls': 0, 'runs': 0, 'wickets': 0,
                'maidens': 0, 'dot_balls': 0,
                'wides': 0, 'no_balls': 0
            }

        self.match_data.innings = 1

        self.update_stats_trees()
        self.update_display()

        # 🔄 Add this line to refresh fielder dropdown
        self.update_fielder_options()           

        # === Start first partnership at innings start ===
        self.match_data.new_batter = self.match_data.non_striker  # Needed for correct partnership creation

        #("✅ Striker:", self.match_data.striker)
        #print("✅ Non-striker:", self.match_data.non_striker)
        self.window.after(200, lambda: self._start_new_partnership({'dismissed_player_id': None}, opening=True))

    def create_center_input_panel(self, parent):
        """Center panel: All input controls including shot details"""
        center_panel = ttk.Frame(parent)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        
        self.input_frame = ttk.LabelFrame(center_panel, text="Ball Input", padding=15)
        self.input_frame.pack(fill=tk.BOTH, expand=True)

        # --- Runs + Blind Turn Section ---
        runs_frame = ttk.Frame(self.input_frame)
        runs_frame.pack(fill=tk.X, pady=5)

        # Left side: Runs
        runs_inner_left = ttk.Frame(runs_frame)
        runs_inner_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(runs_inner_left, text="Runs:").pack(side=tk.LEFT)
        self.runs_var = tk.IntVar(value=0)
        for run in [0, 1, 2, 3, 4, 5, 6]:
            ttk.Radiobutton(runs_inner_left, text=str(run), variable=self.runs_var,
                            value=run, style='info.Toolbutton').pack(side=tk.LEFT, padx=2)

        self.runs_var.trace_add("write", self.update_expected_runs_from_runs)

        self.batter_blind_turn_var = tk.BooleanVar(value=False)
        self.non_striker_blind_turn_var = tk.BooleanVar(value=False)


        # Right side: Blind Turn
        blind_turn_frame = ttk.Frame(runs_frame)
        blind_turn_frame.pack(side=tk.RIGHT, padx=5)

        # Striker Blind Turn
        self.batter_blind_turn_label = ttk.Label(blind_turn_frame, text=f"{self.get_player_name(self.match_data.striker)} Blind Turn:")
        self.batter_blind_turn_label.grid(row=0, column=0, sticky='w', padx=2)

        self.batter_blind_turn_check = ttk.Checkbutton(blind_turn_frame, variable=self.batter_blind_turn_var)
        self.batter_blind_turn_check.grid(row=0, column=1, sticky='w', padx=2)

        # Non-Striker Blind Turn
        self.non_striker_blind_turn_label = ttk.Label(blind_turn_frame, text=f"{self.get_player_name(self.match_data.non_striker)} Blind Turn:")
        self.non_striker_blind_turn_label.grid(row=1, column=0, sticky='w', padx=2)

        self.non_striker_blind_turn_check = ttk.Checkbutton(blind_turn_frame, variable=self.non_striker_blind_turn_var)
        self.non_striker_blind_turn_check.grid(row=1, column=1, sticky='w', padx=2)


        # Extras Section
        extras_frame = ttk.LabelFrame(self.input_frame, text="Extras Details", padding=5)
        extras_frame.pack(fill=tk.X, pady=5)
        
        self.extras_vars = {
            'wides': tk.IntVar(value=0),
            'no_balls': tk.IntVar(value=0),
            'byes': tk.IntVar(value=0),
            'leg_byes': tk.IntVar(value=0),
            'penalty': tk.IntVar(value=0)
        }
        
        extras_grid = ttk.Frame(extras_frame)
        extras_grid.pack(fill=tk.X)
        
        extras_labels = [
            ('Wides', 'wides', 0, 0),
            ('No Balls', 'no_balls', 0, 1),
            ('Byes', 'byes', 0, 2),
            ('Leg Byes', 'leg_byes', 0, 3),
            ('Penalty', 'penalty', 0, 4)
        ]
        
        for label, key, row, col in extras_labels:
            frame = ttk.Frame(extras_grid)
            frame.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            ttk.Spinbox(frame, from_=0, to=6, width=3,
                    textvariable=self.extras_vars[key]).pack(side=tk.LEFT)

        # ====== Enhanced Shot Details ======
        shot_frame = ttk.LabelFrame(self.input_frame, text="Batting Details", padding=5)
        shot_frame.pack(fill=tk.X, pady=5)

        # Footwork Row
        footwork_frame = ttk.Frame(shot_frame)
        footwork_frame.pack(fill=tk.X, pady=2)
        ttk.Label(footwork_frame, text="Footwork:").pack(side=tk.LEFT)
        self.footwork_var = tk.StringVar(value='Nothing')
        footwork_options = ["Nothing", "Front", "Back", "Dance", "Lateral Off", "Lateral Leg", "Sweep"]
        for fwork in footwork_options:
            ttk.Radiobutton(footwork_frame, text=fwork, variable=self.footwork_var,
                        value=fwork, style='warning.Toolbutton').pack(side=tk.LEFT, padx=2)

        # Shot Type Row
        shot_type_frame = ttk.Frame(shot_frame)
        shot_type_frame.pack(fill=tk.X, pady=2)
        ttk.Label(shot_type_frame, text="Shot Type:").pack(side=tk.LEFT)
        self.shot_type_var = tk.StringVar(value='Rotation')
        for stype in ["Attacking", "Defensive", "Rotation"]:
            ttk.Radiobutton(shot_type_frame, text=stype, variable=self.shot_type_var,
                        value=stype, style='info.Toolbutton').pack(side=tk.LEFT, padx=2)

        # Shot Selection Row
        ttk.Label(shot_frame, text="Shot Selection:").pack(anchor=tk.W)
        self.shot_selection_combo = ttk.Combobox(shot_frame, values=[
            'Block', 'Leave', 'Punch', 'Cover Drive', 'Off Drive', 'Straight Drive', 'On Drive', 
            'Square Drive', 'Pull Shot', 'Cut Shot', 'Glance', 'Glide',
            'Reverse Sweep', 'Normal Sweep', 'Slog Sweep', 'Slog', 'Ramp Shot', 'Flick Shot',
            'Late Cut', 'Hook Shot', 'Switch Hit', 'Paddle Sweep', 'Drop and Run', 'Swipe'
        ], state='readonly')
        self.shot_selection_combo.pack(fill=tk.X, pady=2)

        # Aerial/Edged Row
        aerial_frame = ttk.Frame(shot_frame)
        aerial_frame.pack(fill=tk.X, pady=2)
        self.aerial_var = tk.BooleanVar()
        self.clean_hit_var = tk.BooleanVar()
        self.edged_var = tk.BooleanVar()
        self.missed_var = tk.BooleanVar()
        ttk.Checkbutton(aerial_frame, text="Aerial", variable=self.aerial_var,
                    style='danger-round-toggle').pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(aerial_frame, text="Clean Hit", variable=self.clean_hit_var,
                    style='warning-round-toggle').pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(aerial_frame, text="Edged", variable=self.edged_var,
                    style='warning-round-toggle').pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(aerial_frame, text="Missed", variable=self.missed_var,
                        style='danger-round-toggle').pack(side=tk.LEFT, padx=5)
        
        

        # ====== Bowling Details ======
        bowling_frame = ttk.LabelFrame(self.input_frame, text="Bowling Details", padding=5)
        bowling_frame.pack(fill=tk.X, pady=5)

        # Delivery Type
        ttk.Label(bowling_frame, text="Delivery Type:").pack(anchor=tk.W)
        self.delivery_combo = ttk.Combobox(bowling_frame, values=[
            'Outswing', 'Inswing', 'Straight', 'Off Cutter', 'Leg Cutter',
            'Slower Ball', 'Off Spin', 'Leg Spin', 'Quicker Ball'
        ], state='readonly')
        self.delivery_combo.pack(fill=tk.X, pady=2)

        # Over/Around the Wicket toggles
        angle_frame = ttk.Frame(bowling_frame)
        angle_frame.pack(fill=tk.X, pady=5)

        ttk.Label(angle_frame, text="Bowling Angle:").grid(row=0, column=0, sticky=tk.W)

        self.over_var = tk.IntVar(value=1)
        self.around_var = tk.IntVar(value=0)

        def toggle_over():
            if self.over_var.get():
                self.around_var.set(0)
            else:
                self.around_var.set(1)

        def toggle_around():
            if self.around_var.get():
                self.over_var.set(0)
            else:
                self.over_var.set(1)

        ttk.Checkbutton(angle_frame, text="Over the Wicket", variable=self.over_var, command=toggle_over).grid(row=0, column=1, padx=10)
        ttk.Checkbutton(angle_frame, text="Around the Wicket", variable=self.around_var, command=toggle_around).grid(row=0, column=2, padx=10)


        # Fielding Section
        fielding_frame = ttk.LabelFrame(self.input_frame, text="Fielding", padding=5)
        fielding_frame.pack(fill=tk.X, pady=5, anchor=tk.W)

        # Configure grid weights for proper spacing
        # Configure grid columns
        fielding_frame.grid_columnconfigure(0, weight=0)  # Label
        fielding_frame.grid_columnconfigure(1, weight=1)  # Fielder combo
        fielding_frame.grid_columnconfigure(2, weight=0)  # Expected runs
        fielding_frame.grid_columnconfigure(3, weight=0)  # Clear button

        # Fielder Selection (column 0-1)
        ttk.Label(fielding_frame, text="Fielder:").grid(row=0, column=0, sticky='w', padx=0)
        self.fielder_combo = ttk.Combobox(fielding_frame, state='readonly')
        self.fielder_combo.grid(row=0, column=1, padx=5, sticky='ew')

        # Expected Runs (column 2-3)
        ttk.Label(fielding_frame, text="Exp. Runs:").grid(row=0, column=3, sticky='e', padx=5)
        self.expected_runs_var = tk.IntVar(value=0)
        self.expected_runs_spin = ttk.Spinbox(fielding_frame, from_=0, to=10, width=3,
                                            textvariable=self.expected_runs_var)
        self.expected_runs_spin.grid(row=0, column=4, padx=5, sticky='w')

        self.expected_runs_var.trace_add("write", self.on_expected_runs_change)
        
        self.batter_blind_turn_var = tk.BooleanVar()
        self.non_striker_blind_turn_var = tk.BooleanVar()
        



        # Clear button (column 4)
        clear_btn = ttkb.Button(
            fielding_frame,
            text="Clear", 
            bootstyle=(DANGER, OUTLINE),
            command=self.clear_fielding_inputs,
            width=6
        )
        clear_btn.grid(row=0, column=5, padx=5, sticky='e')
        
        
        # Populate fielders from bowling team's playing XI
        bowling_team_players = [
            p for p in self.match_data.selected_players[self.match_data.bowling_team]
            if p not in self.match_data.team1_twelfth + self.match_data.team2_twelfth
        ]
        self.fielder_combo['values'] = [self.get_player_name(p) for p in bowling_team_players]

        # Fielding Events with Boundary Save
        # Fielding Events Label
        ttk.Label(fielding_frame, text="Fielding Events:").grid(row=1, column=0, sticky='w', pady=5)

        # Fielding Events Checkboxes
        check_frame = ttk.Frame(fielding_frame)
        check_frame.grid(row=2, column=0, columnspan=4, sticky='w')
        self.fielding_vars = {
            'Clean Stop/Pick Up': tk.BooleanVar(),
            'Catch': tk.BooleanVar(),
            'Run Out': tk.BooleanVar(),
            'Taken Half Chance': tk.BooleanVar(),
            'Direct Hit': tk.BooleanVar(),
            'Drop Catch': tk.BooleanVar(),
            'Missed Catch': tk.BooleanVar(),
            'Missed Run Out': tk.BooleanVar(),
            'Missed Half Chance': tk.BooleanVar(),
            'Fumble': tk.BooleanVar(),
            'Missed Fielding': tk.BooleanVar(),
            'Overthrow': tk.BooleanVar(),
            'Boundary Save': tk.BooleanVar(),
            'Stumping': tk.BooleanVar(),
            'Missed Stumping': tk.BooleanVar(),
        }
            
        # Organize checkboxes in 3 columns
        for idx, (event_name, var) in enumerate(self.fielding_vars.items()):
            col = idx % 7
            row = idx // 7
            cb = ttk.Checkbutton(check_frame, text=event_name, variable=var)
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            

        # Fielding Style Dropdown
        ttk.Label(fielding_frame, text="Fielding Style:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.fielding_style_combo = ttk.Combobox(fielding_frame, values=['Attacking', 'Defensive', 'Dive', 'Catching', 'WK Normal', 'WK Dive'], state='readonly')
        self.fielding_style_combo.grid(row=3, column=1, columnspan=2, padx=5, sticky='ew', pady=5)
        
        clear_frame = ttkb.Frame(fielding_frame)
        clear_frame.grid(row=0, column=2, rowspan=2, padx=5, sticky='nsew')


            # Configure grid weights
        fielding_frame.columnconfigure(1, weight=1)

        # Dismissal Section - MUST COME BEFORE THE WAGON WHEEL CODE
        dismissal_frame = ttk.Frame(self.input_frame)
        dismissal_frame.pack(fill=tk.X, pady=5)
        
        # Initialize variables FIRST
        self.dismissal_var = tk.BooleanVar(value=False)
        self.dismissal_combo = ttk.Combobox(dismissal_frame, 
                                        values=["Bowled", "Caught", "LBW", "Run Out", "Stumped", "Hit Ball Twice", "Hit Wicket", "Obstructing the Field", "Handled the Ball", "Timed Out"], 
                                        state='disabled')
        
        # Create widgets
        ttk.Checkbutton(dismissal_frame, text="Dismissal", 
                    variable=self.dismissal_var,
                    style='danger.TCheckbutton').pack(side=tk.LEFT)
        self.dismissal_combo.pack(side=tk.LEFT, padx=5)
        
        # Configure trace AFTER variable initialization
        self.dismissal_var.trace_add('write', self._toggle_dismissal_combo)
        
        # THEN add the wagon wheel toggle trace
        self.missed_var.trace_add('write', self.toggle_wagon_wheel)
        self.dismissal_var.trace_add('write', self.toggle_wagon_wheel)

        # --- Action Buttons ---
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.submit_btn = ttk.Button(
            btn_frame,
            text="Submit Ball",
            command=self.record_ball,
            style='success.TButton'
        )
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Undo Last Ball", command=self.undo_ball,
                style='danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Swap Batters", command=self.swap_batters,
                style='warning.TButton').pack(side=tk.LEFT, padx=5)
        
        # Stats Container - Vertical Layout
        stats_container = ttk.Frame(center_panel)
        stats_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Batters Treeview
        batters_frame = ttk.LabelFrame(stats_container, text="Batting Stats", padding=5)
        batters_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Batters Treeview Setup
        self.batters_tree = ttk.Treeview(batters_frame, 
                                    columns=('Name', 'Runs', 'Balls', '4s', '6s', 'SR', 'Status'), 
                                    show='headings', height=3)
        
        # Configure batters columns and headings
        self.batters_tree.heading('Name', text='Name')
        self.batters_tree.heading('Runs', text='Runs')
        self.batters_tree.heading('Balls', text='Balls')
        self.batters_tree.heading('4s', text='4s')
        self.batters_tree.heading('6s', text='6s')
        self.batters_tree.heading('SR', text='SR')
        self.batters_tree.heading('Status', text='Status')
        self.batters_tree.column('Name', width=120)
        self.batters_tree.column('Runs', width=60, anchor='e')
        self.batters_tree.column('Balls', width=60, anchor='e')
        self.batters_tree.column('4s', width=40, anchor='e')
        self.batters_tree.column('6s', width=40, anchor='e')
        self.batters_tree.column('SR', width=60, anchor='e')
        self.batters_tree.column('Status', width=80)
        
        # Batters Scrollbar
        batters_scroll = ttk.Scrollbar(batters_frame)
        batters_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.batters_tree.config(yscrollcommand=batters_scroll.set)
        batters_scroll.config(command=self.batters_tree.yview)
        self.batters_tree.pack(fill=tk.BOTH, expand=True)

        # Bowlers Treeview
        bowlers_frame = ttk.LabelFrame(stats_container, text="Bowling Stats", padding=5)
        bowlers_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Bowlers Treeview Configuration
        self.bowlers_tree = ttk.Treeview(bowlers_frame,
            columns=('Name', 'Overs', 'Dot Balls', 'Runs', 'Wickets', 'Econ', 'Wides', 'No Balls'),
            show='headings',
            height=5
        )
        
        # Set column headings
        self.bowlers_tree.heading('Name', text='Bowler')
        self.bowlers_tree.heading('Overs', text='Overs')
        self.bowlers_tree.heading('Dot Balls', text='Dots')  # Changed from 'Maidens'
        self.bowlers_tree.heading('Runs', text='Runs')
        self.bowlers_tree.heading('Wickets', text='Wkts')
        self.bowlers_tree.heading('Econ', text='Econ')
        self.bowlers_tree.heading('Wides', text='Wd')
        self.bowlers_tree.heading('No Balls', text='Nb')

        # Set column widths
        self.bowlers_tree.column('Name', width=120)
        self.bowlers_tree.column('Overs', width=60, anchor='e')
        self.bowlers_tree.column('Dot Balls', width=60, anchor='e')  # Added this line
        self.bowlers_tree.column('Runs', width=60, anchor='e')
        self.bowlers_tree.column('Wickets', width=50, anchor='e')
        self.bowlers_tree.column('Econ', width=60, anchor='e')
        self.bowlers_tree.column('Wides', width=40, anchor='e')
        self.bowlers_tree.column('No Balls', width=40, anchor='e')
        
        # Bowlers Scrollbar
        bowlers_scroll = ttk.Scrollbar(bowlers_frame)
        bowlers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.bowlers_tree.config(yscrollcommand=bowlers_scroll.set)
        bowlers_scroll.config(command=self.bowlers_tree.yview)
        self.bowlers_tree.pack(fill=tk.BOTH, expand=True)

    def create_left_graphics_panel(self, parent):
        """Left panel with expanded width and detailed graphics"""
        left_panel = ttk.Frame(parent, width=600)  # Increased base width
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights to make panel wider
        parent.grid_columnconfigure(0, weight=120)  # 175 weight for left panel
        parent.grid_columnconfigure(1, weight=100)  # 100 weight for center
        parent.grid_columnconfigure(2, weight=100)  # 100 weight for right
        
        # Pitch Map Frame
        pitch_frame = ttk.LabelFrame(left_panel, text="Pitch Map", padding=10)
        pitch_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.pitch_canvas = tk.Canvas(pitch_frame, bg='white', highlightthickness=0)
        self.pitch_canvas.pack(fill=tk.BOTH, expand=True)
        self.pitch_canvas.bind("<Button-1>", self.record_pitch_location)
        self.pitch_canvas.bind("<Configure>", self.draw_vertical_pitch)
        
        # Wagon Wheel Frame
        wagon_frame = ttk.LabelFrame(left_panel, text="Wagon Wheel", padding=10)
        wagon_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.wagon_canvas = tk.Canvas(wagon_frame, bg='white', highlightthickness=0)
        self.wagon_canvas.pack(fill=tk.BOTH, expand=True)
        self.wagon_canvas.bind("<Button-1>", self.record_shot_location)
        self.wagon_canvas.bind("<Configure>", self.draw_detailed_wagon)

    def on_expected_runs_change(self, *args):
        self.expected_manually_changed = True

    def update_expected_runs_from_runs(self, *args):
        if not self.expected_manually_changed:
            try:
                current_runs = self.runs_var.get()
                self.expected_runs_var.set(current_runs)
            except tk.TclError:
                pass

    def draw_vertical_pitch(self, event=None):
        """Batter's-end oriented pitch map with projection guide"""
        self.pitch_canvas.delete("all")
        w = self.pitch_canvas.winfo_width()
        h = self.pitch_canvas.winfo_height()
        
        # Configuration
        VISIBLE_LENGTH_M = 12.0  # From batter's end (top) down 12 meters
        PITCH_WIDTH_M = 3.0
        STUMP_SPACING_M = 0.3
        CREASE_OFFSET_M = 1.2  # Second crease distance from stumps

        # Calculate scale
        scale = (h * 0.9) / VISIBLE_LENGTH_M
        pitch_width_px = PITCH_WIDTH_M * scale * 1.5  # Extra width for visibility
        pitch_length_px = VISIBLE_LENGTH_M * scale

        # Position at top (batter's end)
        x1 = (w - pitch_width_px) / 2
        y1 = h * 0.08  # 8% from top
        x2 = x1 + pitch_width_px
        y2 = y1 + pitch_length_px

        # Draw pitch surface FIRST
        self.pitch_canvas.create_rectangle(x1, y1, x2, y2, 
                                        outline="#4a3520", fill="#b99d7d", width=3)

        # THEN draw projection zone ON TOP
        stump_center_x = x1 + pitch_width_px/2
        stump_width = STUMP_SPACING_M * scale * 2.5  # Wider projection area
        stump_left = stump_center_x - stump_width/2
        stump_right = stump_center_x + stump_width/2
        
        # Use semi-transparent blue without stipple
        self.pitch_canvas.create_rectangle(stump_left, y1, stump_right, y2,
                                        fill="blue", outline="#2a4e6c", width=2,
                                        tags="projection")

        # Crease lines (white)
        # At stumps (batter's crease)
        self.pitch_canvas.create_line(x1, y1, x2, y1, 
                                    fill="white", width=3, tags="crease")
        
        # 1.2 meters down (bowling crease)
        crease_y = y1 + (CREASE_OFFSET_M * scale)
        self.pitch_canvas.create_line(x1, crease_y, x2, crease_y,
                                    fill="white", width=2, dash=(5,2))

        # Meter markers (0-12m from top)
        for meters in range(0, 13):
            y_pos = y1 + (meters * scale)
            self.pitch_canvas.create_line(x1-15, y_pos, x1, y_pos, 
                                        fill="white", width=1)  # Left marker
            self.pitch_canvas.create_line(x2, y_pos, x2+15, y_pos,
                                        fill="white", width=1)  # Right marker
            
            if meters % 2 == 0:  # Bold labels every 2m
                self.pitch_canvas.create_text(x1-25, y_pos, text=f"{meters}m",
                                            anchor="e", fill="white", 
                                            font=("Arial", 9, "bold"))
                self.pitch_canvas.create_text(x2+25, y_pos, text=f"{meters}m",
                                            anchor="w", fill="white",
                                            font=("Arial", 9, "bold"))

        # Stumps (batter's end at top)
        stump_height = scale * 0.71
        stump_spacing_px = STUMP_SPACING_M * scale
        stump_x = [
            stump_center_x - stump_spacing_px,
            stump_center_x,
            stump_center_x + stump_spacing_px
        ]
        
        for sx in stump_x:
            # Stump poles
            self.pitch_canvas.create_line(sx, y1-stump_height, sx, y1,
                                        fill="white", width=4)
        # Bails
        self.pitch_canvas.create_line(stump_center_x - stump_spacing_px, y1-stump_height, stump_center_x + 10, y1-stump_height,
                                        fill="#ffffff", width=2)
        # Wide lines (0.5m from pitch edge)
        wide_guide_px = -0.5 * scale
        self.pitch_canvas.create_rectangle(x1-wide_guide_px, y1, 
                                        x2+wide_guide_px, y2,
                                        outline="#ffffff", dash=(5,3), width=2)
    
    def draw_detailed_wagon(self, event=None):
        """Detailed circular wagon wheel with oriented pitch"""
        self.wagon_canvas.delete("all")
        w = self.wagon_canvas.winfo_width()
        h = self.wagon_canvas.winfo_height()
        size = min(w, h) * 0.9  # Use 90% of available space
        
        # Circular boundary
        cx = w/2
        cy = h/2
        self.wagon_canvas.create_oval(cx-size/2, cy-size/2, 
                                    cx+size/2, cy+size/2, 
                                    outline="#3d5c1f", width=3)
        
        # Vertical pitch in center
        pitch_length = size * 0.23
        pitch_width = size * 0.07
        self.wagon_canvas.create_rectangle(cx-pitch_width/2, cy-pitch_length/2,
                                        cx+pitch_width/2, cy+pitch_length/2,
                                        fill="#9c7c5b", outline="#5d3c0c")
        
        # Inner ring (e.g., 20m guide)
        inner_radius = size * 0.25
        self.wagon_canvas.create_oval(
            cx - inner_radius, cy - inner_radius,
            cx + inner_radius, cy + inner_radius,
            outline="#bbbbbb", dash=(4, 4)
        )
        
        # Batter's position (top of pitch)
        self.batter_x = cx
        self.batter_y = (cy - pitch_length/2) + 4
        
        # Zone lines (45 degree intervals)
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            x = cx + size/2 * math.cos(math.radians(angle))
            y = cy + size/2 * math.sin(math.radians(angle))
            self.wagon_canvas.create_line(cx, cy, x, y, fill="#666666", dash=(3,3))
        
        # Zone labels
        zones = {
            22: "Mid Wicket",
            67: "Mid On",
            112: "Mid Off",
            157: "Cover",
            202: "Point",
            247: "Third",
            292: "Fine Leg",
            337: "Square Leg"
        }
        for angle, text in zones.items():
            x = cx + (size/2 * 0.8) * math.cos(math.radians(angle))
            y = cy + (size/2 * 0.8) * math.sin(math.radians(angle))
            self.wagon_canvas.create_text(x, y, text=text, fill="#444444")
    
    def toggle_wagon_wheel(self, *args):
        """Disable wagon wheel input for missed balls or certain dismissals"""
        disable_wagon = self.missed_var.get() or \
                       self.dismissal_combo.get() in ['Bowled', 'LBW', 'Stumped']
        
        self.wagon_canvas.config(state=tk.DISABLED if disable_wagon else tk.NORMAL)
        self.wagon_canvas.unbind("<Button-1>")  # Remove existing binding
        
        if not disable_wagon:
            self.wagon_canvas.bind("<Button-1>", self.record_shot_location)
            self.current_shot_location = None
        else:
            self.wagon_canvas.delete("shot_line")

    def _toggle_dismissal_combo(self, *args):
        """Toggle dismissal combo box state"""
        state = 'readonly' if self.dismissal_var.get() else 'disabled'
        self.dismissal_combo.configure(state=state)

    def update_fielder_options(self):
        fielder_team = self.match_data.bowling_team
        fielder_ids = self.match_data.selected_players.get(fielder_team, [])
        fielder_names = [self.get_player_name(pid) for pid in fielder_ids]

        self.fielder_combo["values"] = fielder_names
        if fielder_names:
            self.fielder_combo.current(0)

    def update_blind_turn_labels(self):
        self.batter_blind_turn_label.config(text=f"{self.get_player_name(self.match_data.striker)} Blind Turn:")
        self.non_striker_blind_turn_label.config(text=f"{self.get_player_name(self.match_data.non_striker)} Blind Turn:")

    def record_pitch_location(self, event):
        """Record normalized coordinates with batter's end at top"""
        self.pitch_canvas.delete("pitch_marker")
        w = self.pitch_canvas.winfo_width()
        h = self.pitch_canvas.winfo_height()
        
        # Get pitch dimensions
        VISIBLE_LENGTH_M = 12.0
        scale = (h * 0.9) / VISIBLE_LENGTH_M
        pitch_width_px = 3.0 * scale * 1.5
        pitch_length_px = VISIBLE_LENGTH_M * scale
        
        # Calculate normalized coordinates (0-1 range)
        x1 = (w - pitch_width_px) / 2
        y1 = h * 0.08
        
        norm_x = (event.x - x1) / pitch_width_px
        norm_y = (event.y - y1) / pitch_length_px
        
        # Clamp values and draw marker
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        self.current_pitch_location = (norm_x, norm_y)
        self.pitch_canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5,
                                    outline="#ff0000", width=2, tags="pitch_marker")

    def record_shot_location(self, event):
        """Record shot direction from batter's position"""
        self.wagon_canvas.delete("shot_line")
        
        # Draw line from batter's position to click
        self.wagon_canvas.create_line(self.batter_x, self.batter_y, 
                                    event.x, event.y,
                                    arrow=tk.LAST, fill="#2c5fa8", width=2, 
                                    tags="shot_line")
        
        # Calculate normalized coordinates (0-1) with batter at origin
        dx = event.x - self.batter_x
        dy = event.y - self.batter_y
        max_dist = min(self.wagon_canvas.winfo_width(), 
                    self.wagon_canvas.winfo_height()) / 2
        
        # Normalize to [-1,1] range
        norm_x = dx / max_dist
        norm_y = dy / max_dist
        
        # Convert to polar coordinates for shot classification
        angle = math.degrees(math.atan2(norm_y, norm_x))
        distance = math.hypot(norm_x, norm_y)
        
        self.current_shot_location = {
            'cartesian': (norm_x, norm_y),
            'polar': (angle, distance)
        }

    def create_right_game_panel(self, parent):
        """Right panel: Game situation"""
        right_panel = tk.Frame(parent)
        right_panel.grid(row=0, column=2, sticky="nswe", padx=5)
        
        situation_frame = tk.LabelFrame(right_panel, text="Game Situation", padx=15, pady=15)
        situation_frame.pack(fill=tk.BOTH, expand=True)

        # Score Display
        self.score_label = tk.Label(situation_frame, font=('Helvetica', 24, 'bold'))
        self.score_label.pack(pady=5, fill=tk.X)

        # Overs Display
        self.overs_label = tk.Label(situation_frame, font=('Helvetica', 14))
        self.overs_label.pack(pady=5, fill=tk.X)

        # Phase Indicator (e.g., under target label)
        self.phase_label = ttk.Label(situation_frame, text="Phase: -", font=('Helvetica', 10, 'bold'))
        self.phase_label.pack(anchor="w", padx=10, pady=(2, 0))

        # Run Rate Section
        self.current_rr_var = tk.StringVar(value="0.00")
        self.required_rr_var = tk.StringVar(value="N/A")

        rr_frame = tk.Frame(situation_frame)
        rr_frame.pack(fill=tk.X, pady=5)
        tk.Label(rr_frame, text="Current RR:").grid(row=3, column=0, sticky="w", padx=10)
        tk.Label(rr_frame, textvariable=self.current_rr_var).grid(row=3, column=1, sticky="w", padx=10)

        tk.Label(rr_frame, text="Required RR:").grid(row=4, column=0, sticky="w", padx=10)
        tk.Label(rr_frame, textvariable=self.required_rr_var).grid(row=4, column=1, sticky="w", padx=10)

        self.target_label = tk.Label(rr_frame, text="", font=("Segoe UI", 12, "bold"))
        self.target_label.grid(row=0, column=2, sticky="e", padx=10)        

        # Batters Section
        batters_frame = tk.LabelFrame(situation_frame, text="Batters", pady=10)
        batters_frame.pack(fill=tk.X)
        self.striker_label = tk.Label(batters_frame, font=('Helvetica', 12))
        self.striker_label.pack(anchor='w')
        self.non_striker_label = tk.Label(batters_frame, font=('Helvetica', 12))
        self.non_striker_label.pack(anchor='w')

        # Bowler Section
        bowler_frame = tk.LabelFrame(situation_frame, text="Bowler", pady=10)
        bowler_frame.pack(fill=tk.X)
        self.bowler_label = tk.Label(bowler_frame, font=('Helvetica', 12))
        self.bowler_label.pack(anchor='w')

        # Current Over
        over_frame = tk.LabelFrame(situation_frame, text="Current Over", pady=10)
        over_frame.pack(fill=tk.X)
        self.over_display = tk.Label(over_frame, font=('Helvetica', 12))
        self.over_display.pack(anchor='w')

        self.create_bpi_section(situation_frame)

        # ─────────── Footer Frame (Bottom Right Container) ───────────
        footer_frame = tk.Frame(situation_frame)
        footer_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # ───── End Innings button stays bottom-right
        self.end_innings_button = tk.Button(
            footer_frame,
            text="End Innings",
            bg="red",
            fg="white",
            font=('Helvetica', 10, 'bold'),
            command=self._confirm_end_innings
        )
        self.end_innings_button.pack(anchor='se', padx=5, pady=5)

        # ─────────── Row 1: Rain and Manual End ───────────
        match_action_frame = tk.Frame(footer_frame)
        match_action_frame.pack(anchor='e', padx=5, pady=5)

        ttkb.Button(
            match_action_frame, text="Rain Delay / Adjust Match",
            command=self.handle_rain_adjustment, bootstyle="warning"
        ).pack(side=tk.LEFT, padx=5)

        ttkb.Button(
            match_action_frame, text="End Match (Manual)",
            command=self.manual_end_match_prompt, bootstyle="danger"
        ).pack(side=tk.LEFT, padx=5)

        # ─────────── Row 2: Special Dismissals ───────────
        dismissal_action_frame = tk.Frame(footer_frame)
        dismissal_action_frame.pack(anchor='e', padx=5, pady=5)

        ttkb.Button(
            dismissal_action_frame, text="Non Striker Run Out",
            command=self.handle_mankad, bootstyle="danger"
        ).pack(side=tk.LEFT, padx=5)

        ttkb.Button(
            dismissal_action_frame, text="Retired Not Out",
            command=self.handle_retired_not_out, bootstyle="warning"
        ).pack(side=tk.LEFT, padx=5)

        ttkb.Button(
            dismissal_action_frame, text="Retired Out Now",
            command=self.handle_retired_out, bootstyle="danger"
        ).pack(side=tk.LEFT, padx=5)

        # ─────────── Row 3: Bowler Swap ───────────
        ttkb.Button(
            footer_frame, text="Change Bowler Mid-Over",
            command=self.prompt_bowler_change, bootstyle="warning"
        ).pack(anchor='e', padx=5, pady=5)

    def create_bpi_section(self, parent_frame):
        bpi_frame = tk.LabelFrame(parent_frame, text="Ball Pressure Index", pady=10)
        bpi_frame.pack(fill=tk.X, pady=(10, 0))

        self.batting_bpi_var = tk.StringVar(value="Batting BPI: 0")
        self.bowling_bpi_var = tk.StringVar(value="Bowling BPI: 0")

        self.batting_bpi_label = tk.Label(bpi_frame, textvariable=self.batting_bpi_var, font=('Helvetica', 12))
        self.batting_bpi_label.pack(anchor='w')

        self.bowling_bpi_label = tk.Label(bpi_frame, textvariable=self.bowling_bpi_var, font=('Helvetica', 12))
        self.bowling_bpi_label.pack(anchor='w')

    def update_bpi_display(self, batting_bpi, bowling_bpi):
        self.batting_bpi_var.set(f"Batting BPI: {batting_bpi:.2f}")
        self.bowling_bpi_var.set(f"Bowling BPI: {bowling_bpi:.2f}")

    def handle_rain_adjustment(self):
        popup = tk.Toplevel(self.window)
        popup.title("Rain Delay / Match Adjustment")
        popup.grab_set()

        tk.Label(popup, text="Enter New Max Overs:").pack(pady=5)
        overs_var = tk.IntVar(value=self.match_data.total_overs)
        ttkb.Entry(popup, textvariable=overs_var, width=5).pack()

        target_var = tk.IntVar(value=self.match_data.target_runs or 0)
        if self.match_data.innings == 2:
            tk.Label(popup, text="Enter New Target Score:").pack(pady=5)
            ttkb.Entry(popup, textvariable=target_var, width=5).pack()

        def apply_changes():
            new_overs = overs_var.get()
            new_target = target_var.get() if self.match_data.innings == 2 else None

            # ✅ Update in live match
            self.match_data.total_overs = new_overs
            if new_target is not None:
                self.match_data.target_runs = new_target
            self.match_data.was_rain_delayed = True
            self.match_data.adjusted_overs = new_overs
            self.match_data.overs_phases = define_game_phases(new_overs)

            # ✅ Update in DB
            try:
                conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
                c = conn.cursor()
                c.execute('''
                    UPDATE matches
                    SET rain_interrupted = 1,
                        adjusted_overs = ?,
                        adjusted_target = ?
                    WHERE match_id = ?
                ''', (new_overs, new_target if new_target is not None else 0, self.match_data.match_id))
                conn.commit()
                conn.close()
            except Exception as e:
                messagebox.showerror("Database Error", f"Failed to save rain update:\n{e}")
                return

            self.update_display()
            messagebox.showinfo("Rain Delay Updated", "Match overs/target successfully updated.")
            popup.destroy()

        ttkb.Button(popup, text="Apply", command=apply_changes, bootstyle="success").pack(pady=10)

    def handle_mankad(self):
        # Confirm action
        confirm = messagebox.askyesno("Confirm Mankad", "Was the non-striker run out (Mankad)?")
        if not confirm:
            return

        # Get the dismissed batter (non-striker only)
        dismissed_id = self.match_data.non_striker
        dismissed_name = self.get_player_name(dismissed_id)

        # Mark the batter as dismissed
        self.match_data.wickets += 1
        self.match_data.dismissed_players.add(dismissed_id)
        self.match_data.batters[dismissed_id]['status'] = 'mankad'

        # ✅ Log Mankad dismissal
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute('''
            INSERT INTO non_ball_dismissals (match_id, innings_id, player_id, dismissal_type, over_number)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.match_data.match_id,
            self.match_data.innings_id,
            dismissed_id,
            "Mankad",
            int(self.match_data.current_over)
        ))
        conn.commit()
        conn.close()

        # Prompt for new batter
        available = [
            pid for pid in self.match_data.selected_players[self.match_data.batting_team]
            if pid not in self.match_data.dismissed_players
            and pid not in [self.match_data.striker, self.match_data.non_striker]
        ]

        if not available:
            messagebox.showinfo("All Out", "No available batters left.")
            self.end_innings()
            return

        # Select new batter from popup
        new_window = tk.Toplevel(self.window)
        new_window.title("Select New Batter (Mankad Replacement)")
        new_window.grab_set()

        tk.Label(new_window, text="Select replacement for non-striker:").pack(pady=5)
        self.new_batter = tk.IntVar()
        for pid in available:
            name = self.get_player_name(pid)
            ttk.Radiobutton(new_window, text=name, variable=self.new_batter, value=pid).pack(anchor=tk.W)

        def confirm_selection():
            new_id = self.new_batter.get()
            if not new_id:
                messagebox.showerror("Error", "Select a new batter.")
                return
            new_window.destroy()

            # Update non-striker
            self.match_data.non_striker = new_id
            self.match_data.batters[new_id] = {
                'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'
            }

            # Finalize previous partnership and start new one
            self._finalize_partnership(unbeaten=False)
            self._start_new_partnership(opening=False)

            self.update_display()
            messagebox.showinfo("Mankad Handled", f"{dismissed_name} dismissed (Mankad). New batter added.")

        ttk.Button(new_window, text="Confirm", command=confirm_selection).pack(pady=10)

    def handle_retired_not_out(self):
        retired_window = tk.Toplevel(self.window)
        retired_window.title("Retired Not Out")

        ttk.Label(retired_window, text="Select batter to retire (not out):").pack(pady=5)

        selected = tk.IntVar()

        ttk.Radiobutton(retired_window, text=f"Striker: {self.get_player_name(self.match_data.striker)}", 
                        variable=selected, value=self.match_data.striker).pack(anchor='w', padx=10)

        ttk.Radiobutton(retired_window, text=f"Non-Striker: {self.get_player_name(self.match_data.non_striker)}", 
                        variable=selected, value=self.match_data.non_striker).pack(anchor='w', padx=10)

        def confirm_retire():
            retired_id = selected.get()

            # ✅ Mark as retired not out in match data
            self.match_data.batters[retired_id]['status'] = 'retired not out'
            self.match_data.retired_not_out_players.add(retired_id)

            # ✅ Log in database
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            c.execute('''
                INSERT INTO non_ball_dismissals (match_id, innings_id, player_id, dismissal_type, over_number)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.match_data.match_id,
                self.match_data.innings_id,
                retired_id,
                "Retired Not Out",
                int(self.match_data.current_over)
            ))
            conn.commit()
            conn.close()

            # ✅ Update striker/non-striker logic
            surviving = self.match_data.non_striker if retired_id == self.match_data.striker else self.match_data.striker
            self.match_data.striker = surviving
            self.match_data.non_striker = None  # new batter comes in

            # Prompt for new batter
            available = [
                pid for pid in self.match_data.selected_players[self.match_data.batting_team]
                if pid not in self.match_data.dismissed_players
                and pid not in [self.match_data.striker, self.match_data.non_striker]
                and pid not in self.match_data.team1_twelfth + self.match_data.team2_twelfth
            ]

            if not available:
                messagebox.showinfo("All Out", "No available batters left.")
                self.end_innings()
                return

            new_batter_window = tk.Toplevel(self.window)
            new_batter_window.title("Select New Batter")
            new_batter_window.grab_set()
            self.new_batter = tk.IntVar()

            for pid in available:
                name = self.get_player_name(pid)
                ttk.Radiobutton(new_batter_window, text=name, variable=self.new_batter, value=pid).pack(anchor='w')

            def confirm_new_batter():
                new_batter_window.destroy()
                self.finalize_batter_change(new_batter_window, surviving)
                self.update_display()

            ttk.Button(new_batter_window, text="Confirm", command=confirm_new_batter).pack(pady=10)

            retired_window.destroy()
            self.update_display()

        ttk.Button(retired_window, text="Confirm", command=confirm_retire).pack(pady=10)

    def apply_retired_not_out(self, retired_id):
        retired_name = self.get_player_name(retired_id)

        # Mark as retired in match data
        self.match_data.batters[retired_id]['status'] = 'retired not out'
        self.match_data.dismissed_players.add(retired_id)  # So they are removed from strike

        # Determine who remains
        surviving = self.match_data.non_striker if retired_id == self.match_data.striker else self.match_data.striker

        # Prompt for replacement
        available = [
            pid for pid in self.match_data.selected_players[self.match_data.batting_team]
            if pid not in self.match_data.dismissed_players and pid != surviving
        ]

        if not available:
            messagebox.showinfo("All Out", "No available batters.")
            self.end_innings()
            return

        sel = tk.Toplevel(self.window)
        sel.title("Select New Batter")
        sel.grab_set()
        self.new_batter_var = tk.IntVar()

        for pid in available:
            name = self.get_player_name(pid)
            ttk.Radiobutton(sel, text=name, variable=self.new_batter_var, value=pid).pack(anchor=tk.W)

        def confirm_new_batter():
            new_id = self.new_batter_var.get()
            if not new_id:
                messagebox.showerror("Error", "Select a new batter.")
                return
            sel.destroy()

            # Replace on field
            if retired_id == self.match_data.striker:
                self.match_data.striker = new_id
            else:
                self.match_data.non_striker = new_id

            # Init new batter
            self.match_data.batters[new_id] = {
                'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'
            }

            # Finalize current partnership and start new one
            self._finalize_partnership(unbeaten=False)
            self._start_new_partnership(opening=False)

            self.update_display()
            messagebox.showinfo("Retired", f"{retired_name} marked as Retired Not Out. New batter added.")

        ttk.Button(sel, text="Confirm", command=confirm_new_batter).pack(pady=10)

    def prompt_bowler_change(self):
        popup = tk.Toplevel(self.window)
        popup.title("Change Bowler")
        popup.geometry("300x300")
        popup.grab_set()

        # 🔒 Prevent closing the window with the "X" button
        popup.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning("Required", "Please select a new bowler before continuing."))

        ttk.Label(popup, text="Select New Bowler:").pack(pady=10)

        self.new_bowler_var = tk.IntVar()

        available = [
            p for p in self.match_data.selected_players[self.match_data.bowling_team]
            if p != self.match_data.current_bowler and p not in self.match_data.team1_twelfth + self.match_data.team2_twelfth
        ]

        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        for player_id in available:
            c.execute("SELECT player_name FROM players WHERE player_id = ?", (player_id,))
            name = c.fetchone()[0]
            ttk.Radiobutton(popup, text=name, variable=self.new_bowler_var, value=player_id).pack(anchor="w")
        conn.close()

        ttk.Button(popup, text="Confirm", command=lambda: self.apply_bowler_change(popup)).pack(pady=10)

    def apply_bowler_change(self, popup):
        popup.destroy()
        new_bowler = self.new_bowler_var.get()
        self.match_data.current_bowler = new_bowler

        # Ensure stats entry exists
        if new_bowler not in self.match_data.bowlers:
            self.match_data.bowlers[new_bowler] = {
                'balls': 0, 'runs': 0, 'wickets': 0,
                'dot_balls': 0,
                'wides': 0, 'no_balls': 0
            }

        messagebox.showinfo("Bowler Updated", "New bowler has been set for this over.")
        self.update_display()

    def handle_retired_out(self):
        window = tk.Toplevel(self.window)
        window.title("Retired Out")

        ttk.Label(window, text="Select batter to retire out:").pack(pady=5)

        selected = tk.IntVar()
        ttk.Radiobutton(window, text=f"Striker: {self.get_player_name(self.match_data.striker)}",
                        variable=selected, value=self.match_data.striker).pack(anchor='w', padx=10)
        ttk.Radiobutton(window, text=f"Non-Striker: {self.get_player_name(self.match_data.non_striker)}",
                        variable=selected, value=self.match_data.non_striker).pack(anchor='w', padx=10)

        def confirm_retired_out():
            retired_id = selected.get()
            surviving = self.match_data.non_striker if retired_id == self.match_data.striker else self.match_data.striker

            # Mark retired out
            self.match_data.batters[retired_id]['status'] = 'retired out'
            self.match_data.dismissed_players.add(retired_id)
            self.match_data.dismissed_batter = retired_id

            # Finalize current partnership
            self._finalize_partnership(unbeaten=False)

                # 🔒 Log in non_ball_dismissals table
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()

            c.execute('''
                INSERT INTO non_ball_dismissals (match_id, innings_id, player_id, dismissal_type, over_number)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.match_data.match_id,
                self.match_data.innings_id,
                retired_id,
                "retired_out",
                int(self.match_data.current_over)
            ))

            conn.commit()
            conn.close()

            # Get available new batters
            available = [
                pid for pid in self.match_data.selected_players[self.match_data.batting_team]
                if pid not in self.match_data.dismissed_players
                and pid not in [self.match_data.striker, self.match_data.non_striker]
                and pid not in self.match_data.team1_twelfth + self.match_data.team2_twelfth
            ]

            if not available:
                messagebox.showinfo("All Out", "No available batters left.")
                self.end_innings()
                return

            # Prompt for new batter
            new_batter_window = tk.Toplevel(self.window)
            new_batter_window.title("Select New Batter")
            new_batter_window.grab_set()
            self.new_batter = tk.IntVar()

            for pid in available:
                name = self.get_player_name(pid)
                ttk.Radiobutton(new_batter_window, text=name, variable=self.new_batter, value=pid).pack(anchor='w')

            def confirm_new_batter():
                new_batter_window.destroy()
                self.finalize_batter_change(new_batter_window, surviving)
                self.update_display()

            ttk.Button(new_batter_window, text="Confirm", command=confirm_new_batter).pack(pady=10)

            window.destroy()

        ttk.Button(window, text="Confirm", command=confirm_retired_out).pack(pady=10)

    def prompt_adjust_target(self):
        popup = tk.Toplevel(self.window)
        popup.title("Adjust Target Score")
        popup.grab_set()

        tk.Label(popup, text="Enter Adjusted Target Score (DLS):").pack(pady=10)
        target_var = tk.IntVar(value=self.match_data.target_runs or 0)
        ttkb.Entry(popup, textvariable=target_var, width=5).pack()

        def confirm_target():
            self.match_data.target_runs = target_var.get()
            popup.destroy()

        ttkb.Button(popup, text="Confirm", command=confirm_target, bootstyle="success").pack(pady=10)

    def manual_end_match_prompt(self):
        popup = tk.Toplevel(self.window)
        popup.title("Manually End Match")
        popup.grab_set()

        tk.Label(popup, text="Select Match Result:").pack(pady=5)

        result_var = tk.StringVar()
        margin_var = tk.StringVar()

        options = [
            "Team A won by X runs (DLS)",
            "Team B won by X runs (DLS)",
            "Team A won by X wickets (DLS)",
            "Team B won by X wickets (DLS)",
            "Match Drawn",
            "Match Abandoned"
        ]

        for opt in options:
            ttkb.Radiobutton(popup, text=opt, variable=result_var, value=opt).pack(anchor='w')

        tk.Label(popup, text="Margin (if applicable):").pack(pady=5)
        ttkb.Entry(popup, textvariable=margin_var).pack()

        def confirm():
            result = result_var.get()
            margin = margin_var.get().strip()
            result_text = result.replace("X", margin) if "X" in result else result

            # Determine winner_id based on result text
            team_name = None
            if result.startswith("Team A"):
                team_name = self.match_data.team1 if "won" in result else None
            elif result.startswith("Team B"):
                team_name = self.match_data.team2 if "won" in result else None

            winner_id = None
            if team_name:
                conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
                c = conn.cursor()
                c.execute("SELECT country_id FROM countries WHERE country_name = ?", (team_name,))
                row = c.fetchone()
                if row:
                    winner_id = row[0]
            else:
                conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
                c = conn.cursor()

            # Finalize innings and match
            if self.match_data.current_partnership and self.match_data.current_partnership['balls'] > 0:
                self._finalize_partnership(unbeaten=True)

            c.execute('''
                UPDATE innings
                SET completed = 1
                WHERE innings_id = ?
            ''', (self.match_data.innings_id,))

            c.execute('''
                UPDATE matches
                SET result = ?, rain_interrupted = 1, winner_id = ?
                WHERE match_id = ?
            ''', (result_text, winner_id, self.match_data.match_id))

            conn.commit()
            conn.close()

            messagebox.showinfo("Match Ended", f"Match concluded as: {result_text}")
            self.set_submit_enabled(False)
            popup.destroy()

        # ✅ ADD THIS BUTTON BELOW — OUTSIDE the confirm() definition
        ttkb.Button(popup, text="Confirm", command=confirm, bootstyle=SUCCESS).pack(pady=10)
     
    def clear_fielding_inputs(self):
        """Clear all fielding-related inputs"""
        # Clear fielder selection
        self.fielder_combo.set('')
        
        # Clear fielding style
        self.fielding_style_combo.set('')
        
        # Clear all fielding event checkboxes
        for var in self.fielding_vars.values():
            var.set(False)

    def record_ball(self):
        #print("🚨 record_ball() called")
        if not self.missed_var.get():  # Only validate for non-missed balls
            if not self.validate_non_missed_ball():
                return
        
        # Collect all extras data
        extras = {key: var.get() for key, var in self.extras_vars.items()}

        # Get selected fielding events
        fielding_events = [event for event, var in self.fielding_events.items() if var.get()]

        """Modified to include coordinates validation"""
        if not self.current_pitch_location:
            messagebox.showerror("Error", "Please select pitch location")
            return
        # Check for missed/dismissal first
        if self.missed_var.get() or self.dismissal_combo.get() in ['Bowled', 'LBW', 'Stumped']:
            # Force clear shot data
            self.current_shot_location = None
            self.wagon_canvas.delete("shot_line")
            
            # Bypass shot validation
            ball_data = {
                'shot_coords': (None, None),
                'aerial': False,
                'edged': False,
                'clean_hit': False,
                'runs': 0,
                'extras': extras,
                'dismissal': self.dismissal_var.get(),
                'dismissal_type': self.dismissal_combo.get() if self.dismissal_var.get() else None,
                'pitch_coords': self.current_pitch_location or (None, None),
                'footwork': self.footwork_var.get(),
                'shot_type': self.shot_type_var.get(),
                'shot_selection': self.shot_selection_combo.get(),
                'delivery_type': self.delivery_combo.get(),
                'fielding_events': fielding_events,
                'fielding_style': self.fielding_style_combo.get(),
                'expected_runs': int(self.expected_runs_var.get() or 0),
                'over_the_wicket': self.over_var.get(),
                'around_the_wicket': self.around_var.get()
            }
            self.process_ball(ball_data)
            self.clear_inputs()
            return
        
        # Existing validation for non-missed balls
        if not self.current_shot_location:
            messagebox.showerror("Error", "Please select shot location")
            return



 
        ball_data = {
            'runs': self.runs_var.get(),
            'aerial': self.aerial_var.get(),
            'extras': extras,
            'dismissal': self.dismissal_var.get(),
            'dismissal_type': self.dismissal_combo.get() if self.dismissal_var.get() else None,
            'pitch_coords': self.current_pitch_location or (None, None),
            'shot_coords': self.current_shot_location or (None, None),
            'footwork': self.footwork_var.get(),
            'shot_type': self.shot_type_var.get(),
            'shot_selection': self.shot_selection_combo.get(),
            'edged': self.edged_var.get(),
            'clean_hit': self.clean_hit_var.get(),
            'missed': 0,
            'delivery_type': self.delivery_combo.get(),
            'fielding_events': fielding_events,
            'fielding_style': self.fielding_style_combo.get(),
            'expected_runs': int(self.expected_runs_var.get() or 0),
            'over_the_wicket': self.over_var.get(),
            'around_the_wicket': self.around_var.get()
        }


                # Validate runs input
        try:
            runs = int(self.runs_var.get())
        except ValueError:
            messagebox.showerror("Error", "Runs must be a number!")
            return

        ball_data['runs'] = runs

        # Validate extras
        extras = {}
        for key, var in self.extras_vars.items():
            try:
                value = int(var.get())
            except ValueError:
                messagebox.showerror("Error", f"{key} must be a number!")
                return
            extras[key] = value

        ball_data['extras'] = extras

        # Assign ball_number from match data
        ball_data['ball_number'] = self.match_data.total_balls

        # Convert expected_runs safely
        try:
            ball_data['expected_runs'] = self.expected_runs_var.get()
        except ValueError:
            ball_data['expected_runs'] = 0

        self.process_ball(ball_data)
        self.clear_inputs()

    def validate_non_missed_ball(self):
        """Validate required fields for normal balls"""
        errors = []
        if not self.current_pitch_location:
            errors.append("Please select pitch location")
        # Fix 1: Direct integer validation
        runs = self.runs_var.get()
        if not isinstance(runs, int) or runs not in {0, 1, 2, 3, 4, 5, 6}:
            errors.append("Invalid runs value (must be 0-6)")
        
        # Fix 2: Proper delivery type check
        if not self.delivery_combo.get():
            errors.append("Delivery type required")
            
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        return True

    def select_opening_bowler(self):
        """Select first bowler when interface starts"""
        selector = ttkb.Toplevel(self.window)
        selector.title("Select Opening Bowler")
        selector.geometry("400x400")
        selector.grab_set()

        # 🔒 Prevent closing the window with the "X" button
        selector.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning("Required", "You must select an opening bowler to begin."))

        # Get valid bowlers from playing XI
        bowling_team = self.match_data.bowling_team
        valid_bowlers = [
            p for p in self.match_data.selected_players[bowling_team]
            if p not in (self.match_data.team1_twelfth + self.match_data.team2_twelfth)
        ]

        self.opening_bowler = tk.IntVar(value=valid_bowlers[0])

        for pid in valid_bowlers:
            name = self.get_player_name(pid)
            ttkb.Radiobutton(selector, text=name, variable=self.opening_bowler,
                            value=pid).pack(anchor=tk.W)

        ttkb.Button(selector, text="Start Innings",
                    command=lambda: self.set_opening_bowler(selector)).pack(pady=10)

    def set_opening_bowler(self, window):
        window.destroy()
        selected_id = self.opening_bowler.get()
        
        # Clear existing bowler data
        self.match_data.bowlers = {}
        
        # Initialize new bowler stats
        self.match_data.current_bowler = selected_id
        self.match_data.bowlers[selected_id] = {
            'balls': 0,
            'runs': 0,
            'wickets': 0,
            'maidens': 0,
            'dot_balls': 0,
            'wides': 0,
            'no_balls': 0
        }
        
        self.update_display()

    def process_ball(self, ball_data):
        #print("🚨 process_ball() called")
        #import traceback; traceback.print_stack()
        self.save_state()

                # Calculate total extras
        extras = {
            'wides': self.extras_vars['wides'].get(),
            'no_balls': self.extras_vars['no_balls'].get(),
            'byes': self.extras_vars['byes'].get(),
            'leg_byes': self.extras_vars['leg_byes'].get(),
            'penalty_runs': self.extras_vars['penalty'].get()
        }
        
        ball_data['fielder'] = self.get_fielder_id()

        # Track valid deliveries
        is_valid_delivery = ball_data['extras'].get('no_balls', 0) == 0 and ball_data['extras'].get('wides', 0) == 0

        runs = ball_data['runs']
        bowler = self.match_data.current_bowler

        # Ensure bowler exists in stats
        if bowler not in self.match_data.bowlers:
            self.match_data.bowlers[bowler] = {
                'balls': 0, 'runs': 0, 'wickets': 0,
                'maidens': 0, 'dot_balls': 0,
                'wides': 0, 'no_balls': 0
            }

        # Update bowler stats (only for valid deliveries)
        if is_valid_delivery:
            self.match_data.bowlers[self.match_data.current_bowler]['balls'] += 1
            self.match_data.balls_this_over += 1
        
        # Update batter stats (not for wides/penalty)
        if extras['wides'] == 0:
            self.match_data.batters[self.match_data.striker]['balls'] += 1
            self.match_data.batters[self.match_data.striker]['runs'] += runs
            if runs == 4:
                self.match_data.batters[self.match_data.striker]['fours'] += 1
            elif runs == 6:
                self.match_data.batters[self.match_data.striker]['sixes'] += 1

        if extras['no_balls'] == 1:
            self.match_data.bowlers[self.match_data.current_bowler]['no_balls'] += 1
            ball_data['no_balls'] = 1

        
        if extras['wides'] != 0:
            self.match_data.bowlers[self.match_data.current_bowler]['wides'] += extras['wides']
            ball_data['wides'] = extras['wides']
        
        if extras['byes'] != 0:
            ball_data['byes'] = extras['byes']
        
        if extras['leg_byes'] != 0:
            ball_data['leg_byes'] = extras['leg_byes']

        if extras['penalty_runs'] != 0:
            ball_data['penalty_runs'] = extras['penalty_runs']
        
        # Update total runs
        total_runs = runs + sum(extras.values())
        self.match_data.total_runs += total_runs
        
        # Calculate dot ball status
        is_dot = (is_valid_delivery and ball_data['runs'] == 0 and not any(ball_data['extras'].values()))
        
        # Add to ball data
        ball_data['dot_balls'] = 1 if is_dot else 0

        # Update bowler's conceded runs
        bowler_runs = ball_data['runs'] + extras.get('wides', 0) + extras.get('no_balls', 0)
        self.match_data.bowlers[self.match_data.current_bowler]['runs'] += bowler_runs
        
        
       

        # Always update partnership (even for extras)
        if not ball_data.get('extras', {}).get('wides', 0):
            self.match_data.current_partnership['balls'] += 1
            
            if ball_data['runs'] == 0:
                self.match_data.current_partnership['dots'] += 1
            elif ball_data['runs'] == 1:
                self.match_data.current_partnership['ones'] += 1
            elif ball_data['runs'] == 2:
                self.match_data.current_partnership['twos'] += 1
            elif ball_data['runs'] == 3:
                self.match_data.current_partnership['threes'] += 1
            elif ball_data['runs'] == 4:
                self.match_data.current_partnership['fours'] += 1
            elif ball_data['runs'] == 6:
                self.match_data.current_partnership['sixes'] += 1
                
            self.match_data.current_partnership['runs'] += ball_data['runs']

        # Handle dismissal data
        if ball_data['dismissal']:
            # Show dismissal dialog first
            self.handle_dismissal(ball_data)
            # Assign dismissal data
            ball_data['dismissed_player_id'] = self.dismissed_batter.get()
            ball_data['dismissal_type'] = self.dismissal_combo.get()

        else:
            ball_data['dismissed_player_id'] = None
            ball_data['dismissal_type'] = None


        # Debug output
        #print(f"[DEBUG] Dismissal data - Type: {ball_data.get('dismissal_type')}, Player: {ball_data['dismissed_player_id']}")

        # Handle missed/dismissal cases
        if self.missed_var.get() or ball_data['dismissal_type'] in ['Bowled', 'LBW', 'Stumped']:
            ball_data['shot_coords'] = (None, None)
            ball_data['missed'] = True if self.missed_var.get() else 'Dismissal'

        # Store both data and display symbol - NEW
        ball_symbol = self.get_ball_symbol(ball_data)
        self.match_data.current_over_balls.append({
            'data': ball_data,
            'symbol': ball_symbol
        })

        if extras.get('wides', 0) == 0 and extras.get('no_balls', 0) == 0:
            if ball_data['runs'] == 0:
                self.match_data.bowlers[self.match_data.current_bowler]['dot_balls'] += 1
               
        total_balls = (self.match_data.current_over * 6) + self.match_data.balls_this_over
        max_balls = self.match_data.total_overs * 6
        ball_data["ball_number"] = total_balls

        # Check for innings completion
        if self.match_data.wickets >= 10 or total_balls >= max_balls:
            self.end_innings()
            return
        
        # ✅ Check if delivery was legal (not wide or no ball)
        if extras.get("wides", 0) == 0 and extras.get("no_balls", 0) == 0:

            # ✅ Check for end of over
            if self.match_data.balls_this_over >= 6:
                self.match_data.current_over += 1
                self.match_data.balls_this_over = 0
                self.select_new_bowler()
        
        # Get fielding data
        ball_data['fielding_events'] = [event for event, var in self.fielding_vars.items() if var.get()]
        ball_data['boundary_saved'] = 'Boundary Save' in ball_data['fielding_events']
        ball_data['fielder'] = self.get_fielder_id()  # Your existing fielder selection method

        # 🔍 Calculate expected wicket from fielding mistakes
        ball_data['expected_wicket'] = 0.00
        chance_events = [
            "Drop Catch", "Missed Catch", "Missed Run Out",
            "Missed Half Chance", "Missed Stumping"
        ]

        for event in ball_data['fielding_events']:
            if event in chance_events:
                ew_value = self.prompt_expected_wicket(event)
                if ew_value is not None:
                    ball_data['expected_wicket'] += ew_value
                    #print(f"[DEBUG] Setting the EW value: {ball_data.get('expected_wicket')}")



                # Compute required run rate for second innings
        if self.match_data.innings == 2 and self.match_data.target_runs is not None:
            remaining_runs = self.match_data.target_runs - self.match_data.total_runs
            completed_overs = self.match_data.current_over + (self.match_data.balls_this_over / 6)
            remaining_overs = self.match_data.total_overs - completed_overs
            if remaining_overs > 0:
                ball_data['required_run_rate'] = remaining_runs / remaining_overs
            else:
                ball_data['required_run_rate'] = 0
        else:
            ball_data['required_run_rate'] = 0

        # Determine phase for current ball
        current_over_ph = self.match_data.current_over + 1  # 1-based indexing for display logic

        phases = self.match_data.overs_phases
        pp_start, pp_end = phases['Powerplay']
        mo_start, mo_end = phases['Middle Overs']
        do_start, do_end = phases['Death Overs']

        ball_data['is_powerplay'] = pp_start <= current_over_ph <= pp_end
        ball_data['is_middle_overs'] = mo_start <= current_over_ph <= mo_end
        ball_data['is_death_overs'] = do_start <= current_over_ph <= do_end

        # Retrieve previous ball events and calculate BPI
        ball_events = self.get_previous_ball_events()
        ball_data["partnership_runs"] = self.match_data.current_partnership["runs"]
        
        completed_overs = self.match_data.current_over + (self.match_data.balls_this_over / 6)
        if completed_overs > 0:
            current_rr = self.match_data.total_runs / completed_overs
        else:
            current_rr = 0
        ball_data["current_run_rate"] = round(current_rr, 2)
        
        # Check for new milestone
        partnership_runs = self.match_data.current_partnership["runs"]
        milestones = [
            (25, 0.4),     # +1 bowling pressure, -1 batting pressure
            (50, 0.8),     # +2 bowling pressure, -2 batting pressure
            (75, 1.4),     # +3 bowling pressure, -3 batting pressure
            (100, 2),    # +4 bowling pressure, -4 batting pressure
        ]

        for milestone, weight in milestones:
             if partnership_runs >= milestone and self.match_data.current_partnership.get("last_milestone", 0) < milestone:
                ball_data["partnership_milestone"] = milestone
                self.match_data.current_partnership["last_milestone"] = milestone
                break
        else:
            ball_data["partnership_milestone"] = None

        #print("[DEBUG] Fielding Events:", ball_data.get("fielding_events"))
        #print(f"[DEBUG] dot_balls: {ball_data.get('dot_balls')} | Runs: {ball_data.get('runs')} | Extras: {ball_data.get('extras')}")



        #print("Recent Ball Events Sample:")
        for b in ball_events[-3:]:
            print(f"Ball #{b.get('ball_number')}, Runs: {b.get('runs')}, Expected: {b.get('expected_runs')}")
        batting_bpi, bowling_bpi = self.calculate_bpi(ball_events, ball_data)
        self.update_bpi_display(batting_bpi, bowling_bpi)
        ball_data['batting_bpi'] = batting_bpi
        ball_data['bowling_bpi'] = bowling_bpi
        #print("[DEBUG] Expected Runs (final before save):", ball_data.get("expected_runs"))
        # Set the final striker and non-striker values into ball_data
        ball_data['batter_id'] = self.match_data.striker
        ball_data['non_striker_id'] = self.match_data.non_striker


        ball_data['intent_score'] = self.calculate_batting_intent_score(ball_data)
        ball_id = self.save_ball_event(ball_data)


        ball_data['ball_id'] = ball_id  # ✅ Pass ball_id forward
        self.save_individual_pressure_impact(ball_events, ball_data)

        
         # === Striker Swap Logic (run/extras triggered) ===
        swap = False

        # Check if run-based swap is needed
        if runs in [1, 3, 5]:
            swap = True

        # Check if byes or leg byes are odd
        elif extras.get("byes", 0) in [1, 3, 5] or extras.get("leg_byes", 0) in [1, 3, 5]:
            swap = True

        # Check for wide-based even-number runs (e.g., 2, 4, 6 wides)
        elif extras.get("wides", 0) in [2, 4, 6]:
            swap = True
        
        if self.match_data.balls_this_over == 6:
            swap = True


        if swap:
            self.swap_batters()
            #print("[DEBUG] Batter swapped due to run/extras logic")

        self.update_display()
        self.window.update_idletasks()



        # 🔁 Always check Super Over transition first
        if self.match_data.is_super_over:
            self.check_super_over_transition()
            return
        #print(f"🧪 BEFORE MATCH END CHECK — waiting flag: {self.match_data.waiting_for_new_innings_setup}")

        # ✅ Allow match-ending logic after the first valid ball of 2nd innings
        if self.match_data.waiting_for_new_innings_setup:
            #print("🟢 Second innings setup complete — match checks now enabled.")
            self.match_data.waiting_for_new_innings_setup = False

        # ⛔ Block match-ending logic if second innings hasn't truly started
        if self.match_data.innings == 2 and self.match_data.waiting_for_new_innings_setup:
            #print("⏳ Match-end check skipped — waiting for first real ball in 2nd innings.")
            return




        # 🏏 Regular end-of-match logic
        if self.match_data.innings == 2:
            runs = self.match_data.total_runs
            target = self.match_data.adjusted_target or self.match_data.target_runs

            if runs >= target:
                self.end_match(winner=self.match_data.batting_team)
            elif self.match_data.wickets >= 10 or self.match_data.current_over >= self.match_data.total_overs:
                if runs == target - 1:
                    self.trigger_super_over()
                else:
                    self.end_match(winner=self.match_data.bowling_team)


        self.save_match_state_to_db()

    def end_match(self, winner):
        self.set_submit_enabled(False)

        if self.match_data.current_partnership and self.match_data.current_partnership['balls'] > 0:
            self._finalize_partnership(unbeaten=True)

        batting_team = self.match_data.batting_team
        bowling_team = self.match_data.bowling_team
        runs = self.match_data.total_runs
        target = self.match_data.target_runs
        wickets_lost = self.match_data.wickets

        if runs >= target:
            margin_type = "wickets"
            margin_value = 10 - wickets_lost
            result_text = f"{winner} won by {margin_value} {margin_type}"
        else:
            margin_type = "runs"
            margin_value = target - 1 - runs
            result_text = f"{winner} won by {margin_value} {margin_type}"

        messagebox.showinfo("Match Over", result_text)
        #print(f"🏆 Final Result: {result_text}")

        # ✅ Mark current innings as complete
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute('''
            UPDATE innings
            SET completed = 1
            WHERE innings_id = ?
        ''', (self.match_data.innings_id,))
        
        # ✅ Update match record
        c.execute('''
            UPDATE matches
            SET result = ?, winner_id = ?, margin = ?,
                rain_interrupted = ?, adjusted_overs = ?, adjusted_target = ?
            WHERE match_id = ?
        ''', (
            result_text,
            self.get_team_id(winner),
            margin_value,
            1 if self.match_data.was_rain_delayed else 0,
            self.match_data.total_overs,
            self.match_data.target_runs,
            self.match_data.match_id
        ))

        conn.commit()
        conn.close()

    def get_team_id(self, team_name):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT country_id FROM countries WHERE country_name = ?", (team_name,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

    def prompt_expected_wicket(self, event_name):
        popup = tk.Toplevel(self.window)
        popup.title(f"{event_name} - Chance Difficulty")
        popup.geometry("350x180")
        popup.grab_set()

        ttk.Label(popup, text=f"Rate the difficulty of this chance (1 = easiest, 10 = hardest):",
                font=('Helvetica', 10)).pack(pady=(10, 5))

        difficulty_var = tk.IntVar(value=5)
        expected_label_var = tk.StringVar()

        result = {'value': None}  # This is the key addition to capture the final result

        def update_label(value):
            val = float(value)
            expected = round(1.1 - (val * 0.1), 3)
            expected_label_var.set(f"Expected Wicket Value: {expected:.3f}")

        # Slider
        scale = ttk.Scale(popup, from_=1, to=10, orient=tk.HORIZONTAL,
                        variable=difficulty_var, command=update_label, length=250)
        scale.pack(pady=5)

        # Label for dynamic expected wicket value
        ttk.Label(popup, textvariable=expected_label_var, font=('Helvetica', 11, 'bold')).pack(pady=5)
        update_label(5)  # Initialize label

        def confirm():
            difficulty = float(difficulty_var.get())
            expected = round(1.1 - (difficulty * 0.1), 3)
            result['value'] = expected
            popup.destroy()

        ttk.Button(popup, text="Confirm", command=confirm).pack(pady=10)

        popup.wait_window()  # Wait until popup is closed

        return result['value']  # ✅ Return the value for use in process_ball

    def trigger_super_over(self):
        self.set_submit_enabled(False)

        self.match_data.is_super_over = True
        self.match_data.super_over_round = 1
        self.match_data.super_over_scores = []
        self.match_data.super_over_stage = "TeamA"

        # Prompt user to select which team bats first
        self.prompt_super_over_batting_team()

    def prompt_super_over_batting_team(self):
        popup = tk.Toplevel(self.window)
        popup.title("Select Batting Team for Super Over")
        popup.grab_set()

        tk.Label(popup, text="Who will bat first in the Super Over?").pack(pady=10)

        team_var = tk.StringVar(value=self.match_data.batting_team)  # default

        for team in [self.match_data.batting_team, self.match_data.bowling_team]:
            tk.Radiobutton(popup, text=team, variable=team_var, value=team).pack(anchor="w")

        def confirm_selection():
            selected = team_var.get()
            self.prepare_super_over(first_team=selected)
            popup.destroy()

        ttkb.Button(popup, text="Start Super Over", command=confirm_selection).pack(pady=10)

    def prepare_super_over(self, first_team):
        #print(f"🔥 Preparing Super Over — {first_team} batting")

        # Swap if needed
        second_team = self.match_data.batting_team if self.match_data.batting_team != first_team else self.match_data.bowling_team
        self.match_data.batting_team = first_team
        self.match_data.bowling_team = second_team

        # Set 1 over limit
        self.match_data.total_overs = 1

        # Insert new innings
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Determine new innings number
        c.execute("SELECT MAX(innings) FROM innings WHERE match_id = ?", (self.match_data.match_id,))
        current_max = c.fetchone()[0] or 2
        new_innings_number = current_max + 1

        c.execute('''
            INSERT INTO innings (match_id, innings, batting_team, bowling_team, completed)
            VALUES (?, ?, ?, ?, 0)
        ''', (
            self.match_data.match_id,
            new_innings_number,
            self.match_data.batting_team,
            self.match_data.bowling_team
        ))

        conn.commit()
        self.match_data.innings = new_innings_number
        self.match_data.innings_id = c.lastrowid
        conn.close()

        # Reset match state
        self.match_data.total_runs = 0
        self.match_data.wickets = 0
        self.match_data.current_over = 0
        self.match_data.balls_this_over = 0
        self.match_data.batters = {}
        self.match_data.bowlers = {}
        self.match_data.dismissed_players = set()
        self.match_data.striker = None
        self.match_data.non_striker = None
        self.match_data.new_batter = None
        self.match_data.current_partnership = None
        self.match_data.current_over_balls = []

        self.update_display(reset=True)
        self.clear_stats_trees()
        self.set_submit_enabled(True)
        self.update_fielder_options()

        self.prompt_opening_batters()
        self.select_opening_bowler()

    def check_super_over_transition(self):
        if self.match_data.super_over_stage == "TeamA":
            if self.match_data.wickets >= 2 or self.match_data.current_over >= 1:
                self.team_a_score = self.match_data.total_runs
                self.match_data.super_over_stage = "TeamB"
                self.match_data.batting_team, self.match_data.bowling_team = \
                    self.match_data.bowling_team, self.match_data.batting_team

                messagebox.showinfo("Super Over", "Team B will now bat.")
                self.prepare_super_over(first_team=self.match_data.batting_team)

        elif self.match_data.super_over_stage == "TeamB":
            if self.match_data.wickets >= 2 or self.match_data.current_over >= 1:
                self.team_b_score = self.match_data.total_runs

                # Store round result
                self.match_data.super_over_scores.append((self.team_a_score, self.team_b_score))

                if self.team_a_score != self.team_b_score:
                    self.end_super_over()
                else:
                    # Loop: tied again
                    self.match_data.super_over_round += 1
                    self.match_data.super_over_stage = "TeamA"
                    self.match_data.batting_team, self.match_data.bowling_team = \
                        self.match_data.bowling_team, self.match_data.batting_team

                    messagebox.showinfo("Super Over", f"Another tie! Super Over Round {self.match_data.super_over_round} begins.\nTeam A bats first.")
                    self.prepare_super_over(first_team=self.match_data.batting_team)

    def end_super_over(self):
        self.set_submit_enabled(False)
        last_score = self.match_data.super_over_scores[-1]
        score_a, score_b = last_score

        winner = self.match_data.batting_team if score_b > score_a else self.match_data.bowling_team

        messagebox.showinfo("Match Over", f"{winner} wins the match in Super Over Round {self.match_data.super_over_round}!")
        #print(f"🏏 Super Over Result: {score_a} vs {score_b} → Winner: {winner}")

    def handle_dismissal(self, current_ball):
        # Create dismissal selection window
        dismiss_window = ttkb.Toplevel(self.window)
        dismiss_window.title("Select Dismissed Batter")
        dismiss_window.geometry("400x400")
        dismiss_window.grab_set()
        
        # Get current batter names
        striker_name = self.get_player_name(self.match_data.striker)
        non_striker_name = self.get_player_name(self.match_data.non_striker)
        
        # Radio buttons for batter selection
        self.dismissed_batter = tk.IntVar(value=self.match_data.striker)
        ttkb.Radiobutton(dismiss_window, text=f"Striker: {striker_name}", 
                    variable=self.dismissed_batter, 
                    value=self.match_data.striker).pack(anchor=tk.W)
        ttkb.Radiobutton(dismiss_window, text=f"Non-Striker: {non_striker_name}", 
                    variable=self.dismissed_batter, 
                    value=self.match_data.non_striker).pack(anchor=tk.W)
        
        ttkb.Button(dismiss_window, text="Confirm Dismissal", 
                command=lambda: self.process_dismissal(dismiss_window, current_ball),
                bootstyle=DANGER).pack(pady=10) 

    def calculate_bpi(self, ball_events, current_ball):
        batting_pressure = 0.00
        bowling_pressure = 0.00

        current_runs = current_ball.get("runs", 0)
        expected = current_ball.get("expected_runs", 0)
        dismissal_type = (current_ball.get("dismissal_type") or "").lower()
        batter_shot = current_ball.get("Shot Type", None)
        batter_edge = current_ball.get("edged", False)
        batter_missed = current_ball.get("missed", False)
        is_powerplay = current_ball.get("is_powerplay", False)
        is_middle = current_ball.get("is_middle_overs", False)
        is_death = current_ball.get("is_death_overs", False)
        current_ball_number = current_ball.get("ball_number", None)
        no_ball_extra = current_ball.get("no_balls", None)
        wide_extra = current_ball.get("wides", None)
        leg_bye_extra = current_ball.get("leg_byes", None)
        bye_extra = current_ball.get("byes", None)
        fielding_event = current_ball.get("fielding_events", [])
        partnership = self.match_data.current_partnership
        partnership_runs = current_ball.get("partnership_runs", 0)
        milestone = current_ball.get("partnership_milestone", None)
        last_milestone = partnership.get("last_milestone", 0)

        milestones = [
            (25, 1),
            (50, 2),
            (75, 3),
            (100, 4)
        ]

        debug = {
            "Phase": "",
            "Runs Bonus": 0,
            "Dismissal Bonus": 0,
            "Dot Chain Bonus": 0,
            "Wicket Clump Bonus": 0,
            "Expected vs Actual": 0,
            "Difference Bonus": 0,
            "Run Rate Batting": 0,
            "Run Rate Bowling": 0,
            "Fielding Lingering": [],
            "Boundary Pressure": 0,
            "Extras": [],
            "Shot Type": "",
            "Edge/Miss": [],
            "Fielding Event Impact": [], 
            "Partnership Milestone": None,
            "Wicket Lingering": ""
        }

        # 1. Game phase pressure
        if is_powerplay:
            batting_pressure += 1
            bowling_pressure += 1
            debug["Phase"] = "+1 BP / +1 BoP"
        elif is_middle:
            batting_pressure += 0.5
            bowling_pressure += 0.5
            debug["Phase"] = "+0.5 BP / +0.5 BoP"
        elif is_death:
            batting_pressure += 2
            bowling_pressure += 2
            debug["Phase"] = "+2 BP / +2 BoP"

        # 2. Dismissal bonus
        dismissal_bonus_batter = 0
        dismissal_bonus_bowler = 0
        if dismissal_type:
            if dismissal_type in ["bowled", "lbw", "caught", "stumped"]:
                dismissal_bonus_batter += 1.2
                dismissal_bonus_bowler -= 1.2
                debug["Dismissal Bonus"] = 1.2
            if dismissal_type == "Run Out":
                dismissal_bonus_batter += 2
                dismissal_bonus_bowler -= 2
                debug["Dismissal Bonus"] = 2
            if dismissal_type == "Hit Wicket":
                dismissal_bonus_batter += 1.2
                dismissal_bonus_bowler -= 0.4
                debug["Dismissal Bonus"] = 1.2
        
        if is_powerplay:
            dismissal_bonus_batter *= 1.2
            dismissal_bonus_bowler *= 1.2
        elif is_middle:
            dismissal_bonus_batter *= 0.95
            dismissal_bonus_bowler *= 0.95
        elif is_death:
            dismissal_bonus_batter *= 1.5
            dismissal_bonus_bowler *= 1.5

        batting_pressure += dismissal_bonus_batter
        bowling_pressure += dismissal_bonus_bowler

        # 3. Runs Pressure
        if current_runs == 0 and not any([no_ball_extra, wide_extra, bye_extra, leg_bye_extra]):
            if is_powerplay:
                batting_pressure += 0.2
                bowling_pressure -= 0.2
                debug["Runs Bonus"] = "+0.2 BP / -0.2 BoP (Dot Ball)"
            elif is_middle:
                batting_pressure += 0.125
                bowling_pressure -= 0.125
                debug["Runs Bonus"] = "+0.125 BP / -0.125 BoP (Dot Ball)"
            elif is_death:
                batting_pressure += 0.4
                bowling_pressure -= 0.4
                debug["Runs Bonus"] = "+0.4 BP / -0.4 BoP (Dot Ball)"
            
        elif current_runs == 1:
            batting_pressure -= 0.1
            debug["Runs Bonus"] = "-0.1 BP"
        elif current_runs == 2:
            batting_pressure -= 0.15
            debug["Runs Bonus"] = "-0.15 BP"
        elif current_runs == 3:
            batting_pressure -= 0.2
            debug["Runs Bonus"] = "-0.2 BP"
        elif current_runs == 4:
            batting_pressure -= 0.5
            bowling_pressure += 0.5
            debug["Runs Bonus"] = "-0.5 BP / +0.5 BoP"
        elif current_runs == 5:
            batting_pressure -= 0.75
            bowling_pressure += 0.75
            debug["Runs Bonus"] = "-0.75 BP / +0.75 BoP"
        elif current_runs == 6:
            batting_pressure -= 1
            bowling_pressure += 1
            debug["Runs Bonus"] = "-1 BP / +1 BoP"           


        # 4. Dot ball streak bonus
        dot_streak = 0
        recent_balls = ball_events[-11:] + [current_ball]  # 11 previous + current = 12 total
        for b in reversed(recent_balls):
            if b.get("dot_balls", 0) == 1:
                dot_streak += 1
            else:
                break

        if dot_streak >= 8:
            batting_pressure += 2
            debug["Dot Chain Bonus"] = 2
        elif dot_streak >= 5:
            batting_pressure += 1.5
            debug["Dot Chain Bonus"] = 1.5
        elif dot_streak >= 3:
            batting_pressure += 1
            debug["Dot Chain Bonus"] = 1

        # 4.5 Lingering pressure from recent individual wicket
        last_wicket_ball = None
        for b in reversed(ball_events):
            if b.get("dismissal_type"):
                last_wicket_ball = b.get("ball_number")
                break
        
        
        #print(f"[DEBUG] Last Wicket Ball: {last_wicket_ball}")
        #print(f"[DEBUG] Current Ball Number: {current_ball_number}")

        if last_wicket_ball is not None and current_ball_number is not None:
            balls_since_wicket = current_ball_number - last_wicket_ball
            if 0 < balls_since_wicket <= 6:
                if balls_since_wicket == 1:
                    batting_pressure += 1.5
                    bowling_pressure -= 0.5
                    debug["Wicket Lingering"] = "True"
                elif balls_since_wicket == 2:
                    batting_pressure += 1.25
                    bowling_pressure -= 0.42
                elif balls_since_wicket == 3:
                    batting_pressure += 1
                    bowling_pressure -= 0.34
                elif balls_since_wicket == 4:
                    batting_pressure += 0.75
                    bowling_pressure -= 0.26
                elif balls_since_wicket == 5:
                    batting_pressure += 0.5
                    bowling_pressure -= 0.18
                elif balls_since_wicket == 6:
                    batting_pressure += 0.5
                    bowling_pressure -= 0.1
            else:
                print("[DEBUG] Wicket lingering not applied — either no previous wicket or current ball is None")

        # 5. Extras Pressure
        if no_ball_extra and no_ball_extra > 0:
            batting_pressure -= 0.6
            bowling_pressure += 0.4
            debug["Extras"].append("No Ball")
        if wide_extra and wide_extra > 0:
            batting_pressure -= 0.15
            bowling_pressure += 0.05
            debug["Extras"].append("Wide")
        if bye_extra and bye_extra > 0:
            batting_pressure -= 0.1
            bowling_pressure += 0.1
            debug["Extras"].append("Bye")
        if leg_bye_extra and leg_bye_extra > 0:
            batting_pressure -= 0.1
            debug["Extras"].append("Leg Bye")

        #print(f"[DEBUG] Shot Type: {batter_shot}")

        # 6. Batting Pressure based on input
        if batter_shot == "Aggressive":
            bowling_pressure += 0.05
            debug["Shot Type"] = batter_shot
            #print("[DEBUG] Aggressive shot detected — +0.05 Bowling Pressure")
        if batter_edge == 1:
            batting_pressure += 0.1
            debug["Edge/Miss"].append("Edge")
        if batter_missed == 1:
            batting_pressure += 0.1
            debug["Edge/Miss"].append("Miss")


        # 7. Wicket clump logic (2 wickets in 2 overs, 4 wickets in 6 overs)
        recent_wickets = [
            b for b in ball_events[-36:] if b.get("dismissal_type")
        ]

        if len(recent_wickets) >= 4:
            batting_pressure += 2
            debug["Wicket Clump Bonus"] = 2
        elif len(recent_wickets) >= 2:
            batting_pressure += 1
            debug["Wicket Clump Bonus"] = 1

        #print("[DEBUG] Fielding Events:", current_ball.get("fielding_events", []))

        # 8. Fielding Events Pressure
        if "Drop Catch" in fielding_event:
            batting_pressure -= 0.5
            bowling_pressure += 1
            debug["Fielding Event Impact"].append("Drop Catch")
        if "Missed Catch" in fielding_event:
            batting_pressure -= 0.5
            bowling_pressure += 1
            debug["Fielding Event Impact"].append("Missed Catch")
        if "Direct Hit" in fielding_event:
            batting_pressure += 0.2
            debug["Fielding Event Impact"].append("Direct Hit")
        if "Missed Run Out" in fielding_event:
            batting_pressure -= 0.3
            bowling_pressure += 0.5
            debug["Fielding Event Impact"].append("Missed Run Out")
        if "Missed Fielding" in fielding_event:
            batting_pressure -= 0.2
            bowling_pressure += 0.3
            debug["Fielding Event Impact"].append("Missed Fielding")
        if "Boundary Save" in fielding_event:
            batting_pressure += 0.25
            bowling_pressure -= 0.05
            debug["Fielding Event Impact"].append("Boundary Save")
        if "Clean Stop/Pick Up" in fielding_event:
            batting_pressure += 0.02
            bowling_pressure -= 0.02
            debug["Fielding Event Impact"].append("Clean Stop/Pick Up")
        if "Fumble" in fielding_event:
            batting_pressure -= 0.1
            bowling_pressure += 0.15
            debug["Fielding Event Impact"].append("Fumble")
        if "Overthrow" in fielding_event:
            batting_pressure -= 0.1
            bowling_pressure += 0.1
            debug["Fielding Event Impact"].append("Overthrow")



        # 9. Expected vs Actual Runs (Fielding error or save)
        delta = expected - current_runs
        debug["Expected vs Actual"] = delta
        if delta > 0:
            # Bowler saved runs
            batting_pressure += 0.4
            bowling_pressure -= 0.2
            debug["Difference Bonus"] = "+0.4 BP / -0.2 BoP"
        elif delta < 0:
            # Fielder error (allowed extra)
            batting_pressure -= 0.4
            bowling_pressure += 0.2
            debug["Difference Bonus"] = "-0.4 BP / +0.2 BoP"
        elif delta == 0:
            # Nothing
            batting_pressure += 0
            bowling_pressure += 0
            debug["Difference Bonus"] = "0 BP / 0 BoP"

        # 10. Fielding event lingering effects
        lingering_effects = {
            "drop catch": 12,
            "missed fielding": 6,
            "missed run out": 6
        }

        for b in ball_events[-18:]:
            past_ball_num = b.get("ball_number", 0)
            for event in b.get("fielding_events", []):
                event_lower = event.lower()
                if event_lower in lingering_effects and current_ball_number is not None:
                    if current_ball_number - past_ball_num <= lingering_effects[event_lower]:
                        batting_pressure -= 0.5
                        bowling_pressure += 1
                        debug["Fielding Lingering"].append(f"{event} on ball {past_ball_num}")



        # 11. Run Rate Pressure (Innings Contextual) ======

        innings = current_ball.get("innings", 1)
        current_rr = current_ball.get("current_run_rate", 0)
        required_rr = current_ball.get("required_run_rate", 0)
        expected_rr = 6  # You can adjust this baseline for 1st innings
        #print(f"[DEBUG] Innings: {innings}, Required RR: {required_rr}, Current RR: {current_rr}")

        if innings == 2 and required_rr > 0:
            rr_diff = required_rr - current_rr
            #print(f"[DEBUG] RR Diff: {rr_diff}")
            if rr_diff >= 2:
                batting_pressure += 2
                bowling_pressure -= 1
                debug["Run Rate Batting"] = "2Batting +2, Bowling -1"
                debug["Run Rate Bowling"] = "2Batting +2, Bowling -1"
            elif rr_diff >= 1:
                batting_pressure += 1
                bowling_pressure -= 0.5
                debug["Run Rate Batting"] = "2Batting +1, Bowling -0.5"
                debug["Run Rate Bowling"] = "2Batting +1, Bowling -0.5"
            elif rr_diff >= 0:
                batting_pressure += 0.5
                bowling_pressure -= 0.25
                debug["Run Rate Batting"] = "2Batting +0.5, Bowling -0.25"
                debug["Run Rate Bowling"] = "2Batting +0.5, Bowling -0.25"
            elif rr_diff <= -2:
                batting_pressure -= 1
                bowling_pressure += 2
                debug["Run Rate Batting"] = "2Batting -1, Bowling +2"
                debug["Run Rate Bowling"] = "2Batting -1, Bowling +2"
            elif rr_diff <= -1:
                batting_pressure -= 0.5
                bowling_pressure += 1
                debug["Run Rate Batting"] = "2Batting -0.5, Bowling +1"
                debug["Run Rate Bowling"] = "2Batting -0.5, Bowling +1"
            elif rr_diff <= -0.01:
                batting_pressure -= 0.25
                bowling_pressure += 0.5
                debug["Run Rate Batting"] = "2Batting -0.25, Bowling +0.5"
                debug["Run Rate Bowling"] = "2Batting -0.25, Bowling +0.5"
        elif innings == 1:
            rr_diff = expected_rr - current_rr
            #print(f"[DEBUG] RR Diff: {rr_diff}")
            if rr_diff >= 2:
                batting_pressure += 2
                bowling_pressure -= 1
                debug["Run Rate Batting"] = "Batting +2, Bowling -1"
                debug["Run Rate Bowling"] = "Batting +2, Bowling -1"
            elif rr_diff >= 1:
                batting_pressure += 1
                bowling_pressure -= 0.5
                debug["Run Rate Batting"] = "Batting +1, Bowling -0.5"
                debug["Run Rate Bowling"] = "Batting +1, Bowling -0.5"
            elif rr_diff >= 0:
                batting_pressure += 0.5
                bowling_pressure -= 0.25
                debug["Run Rate Batting"] = "Batting +0.5, Bowling -0.25"
                debug["Run Rate Bowling"] = "Batting +0.5, Bowling -0.25"
            elif rr_diff <= -2:
                batting_pressure -= 1
                bowling_pressure += 2
                debug["Run Rate Batting"] = "Batting -1, Bowling +2"
                debug["Run Rate Bowling"] = "Batting -1, Bowling +2"
            elif rr_diff <= -1:
                batting_pressure -= 0.5
                bowling_pressure += 1
                debug["Run Rate Batting"] = "Batting -0.5, Bowling +1"
                debug["Run Rate Bowling"] = "Batting -0.5, Bowling +1"
            elif rr_diff <= -0.01:
                batting_pressure -= 0.25
                bowling_pressure += 0.5
                debug["Run Rate Batting"] = "Batting -0.25, Bowling +0.5"
                debug["Run Rate Bowling"] = "Batting -0.25, Bowling +0.5"


        # 9. Boundary pressure (persistent for whole over if early boundary exists)
        current_ball_number = current_ball.get("ball_number", 0)
        current_over = current_ball_number // 6

        early_boundary_this_over = False
        for b in ball_events:
            past_ball_number = b.get("ball_number", 0)
            past_over = past_ball_number // 6
            ball_in_over = past_ball_number % 6
            if past_over == current_over and b.get("runs", 0) >= 4 and ball_in_over in [1, 2]:
                early_boundary_this_over = True
                break

        # Check if current ball is the early boundary
        ball_in_over_now = current_ball_number % 6
        if current_runs >= 4 and ball_in_over_now in [1, 2]:
            early_boundary_this_over = True

        if early_boundary_this_over:
            bowling_pressure += 1
            batting_pressure -= 1
            debug["Boundary Pressure"] = "-1 BP / +1 BoP (Early Over Boundary Effect)"
        else:
            debug["Boundary Pressure"] = "+0"


        for milestone, weight in milestones:
            if partnership_runs >= milestone > last_milestone:
                batting_pressure -= weight
                bowling_pressure += weight
                partnership["last_milestone"] = milestone  # Update so it's only counted once
                if debug.get("Partnership Milestone"):
                    milestone, weight = debug["Partnership Milestone"]
                    #print(f"Partnership Milestone Triggered: {milestone} runs with weight {weight}")
                    
                break  # Only apply one milestone per ball


        #print(f"\n=== TEAM BPI DEBUG — Ball {current_ball.get('ball_number')} ===")
        #print(f"Phase Adjustment             : {debug['Phase']}")
        #print(f"Runs Pressure               : {debug['Runs Bonus']}")
        #print(f"Dismissal Bonus              : +{debug['Dismissal Bonus']} ({dismissal_type})")
        #print(f"Dot Chain Bonus              : +{debug['Dot Chain Bonus']}")
        #print(f"Wicket Clump Bonus           : +{debug['Wicket Clump Bonus']} ({len(recent_wickets)} wickets in last 6 overs)")
        #print(f"Expected vs Actual Runs      : Expected {expected}, Actual {current_runs}, Δ: {debug['Expected vs Actual']}")
        #print(f"Run Rate Pressure — Batting : {debug['Run Rate Batting']}")
        #print(f"Run Rate Pressure — Bowling : {debug['Run Rate Bowling']}")
        #print(f"Fielding Event Impact        : {debug['Fielding Lingering']}")
        #print(f"Boundary Pressure            : +{debug['Boundary Pressure']}")
        #if "Partnership Milestone" in debug:
        #    if debug.get("Partnership Milestone"):
        #            milestone, weight = debug["Partnership Milestone"]
        #            print(f"Partnership Milestone Triggered: {milestone} runs with weight {weight}")
        #print(f"Final BPI — Batting          : {batting_pressure}")
        #print(f"Final BPI — Bowling          : {bowling_pressure}")
        #print(f"Wicket Lingering Effect      : {debug['Wicket Lingering']}")
        #print("=" * 30)

        return batting_pressure, bowling_pressure

    def save_individual_pressure_impact(self, ball_events, current_ball):
        ball_id = current_ball.get('ball_id')
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        impacts = []
        
        current_runs = current_ball.get("runs", 0)
        expected = current_ball.get("expected_runs", 0)
        dismissal_type = (current_ball.get("dismissal_type") or "").lower()
        batter_shot = current_ball.get("shot_type", None)
        batter_edge = current_ball.get("edged", False)
        batter_missed = current_ball.get("missed", False)
        current_ball_number = current_ball.get("ball_number", None)
        no_ball_extra = current_ball.get("no_balls", None)
        wide_extra = current_ball.get("wides", None)
        bye_extra = current_ball.get("byes", None)
        fielding_events = current_ball.get("fielding_events", [])
        fielding_style = current_ball.get("fielding_style", None)
        partnership = self.match_data.current_partnership
        partnership_runs = partnership.get("runs", 0)
        last_milestone = partnership.get("last_milestone", 0)

        milestones = [
            (25, 1),
            (50, 2),
            (75, 3),
            (100, 4)
        ]


        striker = self.match_data.striker
        non_striker = self.match_data.non_striker
        bowler = self.match_data.current_bowler
        fielder = current_ball.get("fielder")


        if dismissal_type:
            if dismissal_type in ["bowled", "lbw"]:
                impacts.append((striker, "batting", "pressure_applied", -1.2, "Dismissed"))
                impacts.append((bowler, "bowling", "pressure_applied", +1.2, "Wicket"))
            if dismissal_type == "caught":
                impacts.append((striker, "batting", "pressure_applied", -1.2, "Dismissed"))
                impacts.append((bowler, "bowling", "pressure_applied", +0.4, "Wicket"))
                impacts.append((fielder, "fielding", "pressure_applied", +0.8, "Catch")) 
            if dismissal_type == "stumped":
                impacts.append((striker, "batting", "pressure_applied", -1.2, "Dismissed"))
                impacts.append((bowler, "bowling", "pressure_applied", +0.4, "Wicket"))
                impacts.append((fielder, "fielding", "pressure_relieved", +0.8, "Stumping"))
            if dismissal_type == "run out":
                impacts.append((striker, "batting", "pressure_applied", -1.2, "Dismissed"))
                impacts.append((fielder, "fielding", "pressure_applied", +2.0, "Run Out"))
            if dismissal_type == "hit wicket":
                impacts.append((striker, "batting", "pressure_applied", -1.2, "Hit Wicket"))
                impacts.append((bowler, "bowling", "pressure_applied", +0.4, "Credit (Hit Wicket)"))

        # --- Runs Pressure ---
        if current_runs == 0:
            impacts.append((striker, "batting", "pressure_applied", -0.2, "Dot Ball"))
            impacts.append((bowler, "bowling", "pressure_applied", +0.2, "Dot Ball"))
        elif current_runs == 1:
            impacts.append((striker, "batting", "pressure_relieved", +0.1, "Single"))
        elif current_runs == 2:
            impacts.append((striker, "batting", "pressure_relieved", +0.2, "Double"))
        elif current_runs == 3:
            impacts.append((striker, "batting", "pressure_relieved", +0.3, "Triple"))
        elif current_runs == 4:
            impacts.append((striker, "batting", "pressure_relieved", +0.5, "Boundary"))
            impacts.append((bowler, "bowling", "pressure_relieved", -0.5, "Boundary"))
        elif current_runs == 5:
            impacts.append((striker, "batting", "pressure_relieved", +0.75, "Five"))
            impacts.append((bowler, "bowling", "pressure_relieved", -0.75, "Five"))
        elif current_runs == 6:
            impacts.append((striker, "batting", "pressure_relieved", +1.0, "Six"))
            impacts.append((bowler, "bowling", "pressure_relieved", -1.0, "Six"))         


        # --- Dot Chain Pressure ---
        dot_streak = 0
        recent_balls = ball_events[-11:] + [current_ball] 
        for b in reversed(recent_balls):
            if b.get("dot_balls", 0) == 1:
                dot_streak += 1
            else:
                break

        if dot_streak >= 8:
            impacts.append((striker, "batting", "pressure_applied", -2.0, "8 Dots"))
            impacts.append((bowler, "bowling", "pressure_applied", +2.0, "8 Dots"))
        elif dot_streak >= 5:
            impacts.append((striker, "batting", "pressure_applied", -1.5, "5 Dots"))
            impacts.append((bowler, "bowling", "pressure_applied", +1.5, "5 Dots"))
        elif dot_streak >= 3:
            impacts.append((striker, "batting", "pressure_applied", -1.0, "3 Dots"))
            impacts.append((bowler, "bowling", "pressure_applied", +1.0, "3 Dots"))


        # --- Extras ---
        if no_ball_extra:
            impacts.append((bowler, "bowling", "pressure_relieved", -0.4, "No Ball"))
        if wide_extra:
            impacts.append((bowler, "bowling", "pressure_relieved", -0.15, "Wide"))
        if bye_extra:
            impacts.append((fielder, "fielding", "pressure_relieved", -0.1, "Bye"))


        # --- Shot and Outcome ---
        if batter_shot == "Aggressive":
            impacts.append((striker, "batting", "pressure_relieved", +0.05, "Aggressive Shot"))
        if batter_edge:
            impacts.append((striker, "batting", "pressure_applied", -0.1, "Edged"))
        if batter_missed:
            impacts.append((striker, "batting", "pressure_applied", -0.1, "Missed"))

        # --- Fielding Events ---
        event_map = {
            "drop catch": (-1.0, "Drop Catch"),
            "missed catch": (-1.0, "Missed Catch"),
            "direct hit": (+0.2, "Direct Hit"),
            "missed run out": (-0.5, "Missed Run Out"),
            "missed fielding": (-0.15, "Missed Fielding"),
            "boundary save": (+0.25, "Boundary Save"),
            "clean stop/pick up": (+0.05, "Clean Stop"),
            "fumble": (-0.1, "Fumble"),
            "overthrow": (-0.1, "Overthrow")
        }
        for event in fielding_events:
            if fielder and event in event_map:
                val, reason = event_map[event]
                impacts.append((fielder, "fielding", "pressure_relieved" if val > 0 else "pressure_applied", val, reason))

        # --- Expected vs Actual Runs ---
        delta = expected - current_runs
        if delta > 0 and fielder:
            impacts.append((fielder, "fielding", "pressure_applied", +0.4, "Run Saving"))
        if delta < 0 and fielder:
            if "Clean Stop/Pick Up" in fielding_events and fielding_style == "Attacking":
                impacts.append((striker, "batting", "pressure_relieved", +0.4, "Tight Run"))
            else:
                impacts.append((fielder, "fielding", "pressure_relieved", -0.4, "Run Conceded"))

            



        # --- Early Over Boundary Boost ---
        if current_runs >= 4 and current_ball_number % 6 in [0, 1]:
            impacts.append((striker, "batting", "pressure_relieved", +1.0, "Early Over Boundary"))
            impacts.append((bowler, "bowling", "pressure_relieved", -1.0, "Early Over Boundary"))


        milestones = [(25, 1), (50, 2), (75, 3), (100, 4)]
        last_milestone = self.match_data.current_partnership.get("last_milestone", 0)

        for milestone, weight in milestones:
            if partnership_runs >= milestone > last_milestone:
                impacts.append((striker, "batting", "pressure_relieved", +0.5 * weight, "Partnership Striker"))
                impacts.append((non_striker, "batting", "pressure_relieved", +0.5 * weight, "Partnership Non-striker"))
                self.match_data.current_partnership["last_milestone"] = milestone
                #print(f"[DEBUG] Partnership Milestone Reached: {milestone} runs — Batting -{weight}, Bowling +{weight}")
                break  # only process one milestone per ball

        print(f"\n=== INDIVIDUAL PRESSURE IMPACT — Ball {current_ball_number} ===")
        for player_id, team_role, pressure_type, pressure_value, reason in impacts:
            print(f"{team_role.title()} (ID {player_id}) : {pressure_type} {pressure_value} ({reason})")
        print("=" * 30)

        # --- Save to DB ---
        for player_id, team_role, pressure_type, pressure_value, reason in impacts:
            if player_id:
                c.execute('''
                    INSERT INTO player_pressure_impact (ball_id, player_id, team_role, pressure_type, pressure_value, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ball_id, player_id, team_role, pressure_type, pressure_value, reason))

        conn.commit()
        conn.close()

    def calculate_batting_intent_score(self, ball_data, previous_ball=None):
        SHOT_TYPE_BASE = {
            "Attacking": 5,
            "Rotation": 3,
            "Defensive": 1
        }

        SHOT_INTENT_LOOKUP = {
            "Block": 0, "Leave": 0, "Punch": 2, "Cover Drive": 3, "Off Drive": 3,
            "Straight Drive": 3, "On Drive": 3, "Square Drive": 3, "Pull Shot": 4,
            "Cut Shot": 3, "Glance": 2, "Glide": 2, "Reverse Sweep": 5, "Normal Sweep": 4,
            "Slog Sweep": 5, "Slog": 5, "Ramp Shot": 5, "Flick Shot": 3, "Late Cut": 2,
            "Hook Shot": 4, "Switch Hit": 5, "Paddle Sweep": 4, "Drop and Run": 3, "Swipe": 4
        }

        FOOT_INTENT_LOOKUP = {
            "Nothing": 0, "Back": 1, "Front": 2, "Lateral Off": 2,
            "Lateral Leg": 2, "Sweep": 3, "Dance": 5
        }

        # FIX: Match actual key names used in ball_data
        shot_type = ball_data.get("shot_type", "")
        shot_name = ball_data.get("shot_selection", "")  # Was incorrectly using 'shot_name'
        footwork = ball_data.get("footwork", "")

        shot_type_score = SHOT_TYPE_BASE.get(shot_type, 3)
        shot_name_score = SHOT_INTENT_LOOKUP.get(shot_name, 2)
        footwork_score = FOOT_INTENT_LOOKUP.get(footwork, 0)

        raw_score = shot_type_score + shot_name_score + footwork_score

        actual = ball_data.get("runs", 0)
        expected = ball_data.get("expected_runs", 0)
        fielding_style = ball_data.get("fielding_style", "").lower()
        print(f"🧠 INTENT CONTEXT DEBUG → Actual: {actual} | Expected: {expected}")

        # FIX: Fielding event was missing; default safely or expand in future
        fielding_event = ball_data.get("fielding_events")
        print(f"🧠 FIELDING EVENT: {fielding_event}")
        if actual > expected and "Clean Stop/Pick Up" in fielding_event and fielding_style == "attacking":
            print("✅ Tight running bonus applied")
            raw_score += 4

        # Bonus: Boundary hit
        if actual >= 4:
            raw_score += 2
            print("✅ Boundary bonus applied (+1)")

        if int(ball_data.get("aerial", 0)) == 1:
            raw_score += 1

        if previous_ball:
            if previous_ball.get("shot_type") == "Aggressive" and shot_type == "Aggressive":
                raw_score += 1

        raw_score = min(raw_score, 15)
        normalized_score = min(round((raw_score / 12) * 10, 2), 10)

        print(f"🧠 Intent Score Debug — Shot Type: {shot_type}, Shot: {shot_name}, Footwork: {footwork} → Score: {normalized_score}")

        return normalized_score

    def get_last_ball_id(self): 
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT MAX(ball_id) FROM ball_events")
        result = c.fetchone()
        conn.close()
        return result[0] if result and result[0] else None
    
    def process_dismissal(self, window, current_ball):
        window.destroy()
        dismissed_id = self.dismissed_batter.get()

        # Update match data
        self.match_data.wickets += 1

        # Only credit bowler if it’s a valid dismissal
        bowler_credit_dismissals = {
            'bowled', 'caught', 'lbw', 'stumped', 'hit wicket', 'hit the ball twice'
        }
        dismissal_type = (current_ball.get("dismissal_type") or "").lower()

        if dismissal_type in bowler_credit_dismissals:
            self.match_data.bowlers[self.match_data.current_bowler]['wickets'] += 1
            
        self.match_data.batters[dismissed_id]['status'] = 'out'
        self.match_data.dismissed_players.add(dismissed_id)

        # Finalize old partnership
        partnership_to_finalize = self.match_data.current_partnership.copy()
        self._finalize_partnership(unbeaten=False, previous_partnership=partnership_to_finalize)

        # Determine surviving batter
        surviving_batter = (self.match_data.striker if dismissed_id == self.match_data.non_striker 
                            else self.match_data.non_striker)

        # Get available new batters
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        excluded_ids = [surviving_batter] + list(self.match_data.dismissed_players)
        twelfth_ids = self.match_data.team1_twelfth + self.match_data.team2_twelfth
        playing_xi = self.match_data.selected_players[self.match_data.batting_team]

        query = f'''SELECT player_id, player_name FROM players 
                    WHERE country_id = (
                        SELECT country_id FROM countries 
                        WHERE country_name = ?
                    )
                    AND player_id IN ({','.join(['?']*len(playing_xi))})
                    AND player_id NOT IN ({','.join(['?']*len(excluded_ids))})
                    AND player_id NOT IN ({','.join(['?']*len(twelfth_ids))})'''

        params = (self.match_data.batting_team,) + tuple(playing_xi) + tuple(excluded_ids) + tuple(twelfth_ids)

        c.execute(query, params)
        available = c.fetchall()
        conn.close()

        if not available:
            possibly_returnable = [
                pid for pid in self.match_data.retired_not_out_players
                if pid not in self.match_data.dismissed_players
            ]

            if possibly_returnable:
                if messagebox.askyesno(
                    "Retired Batter Available",
                    "A retired not out batter is available. Do you want to end the innings or bring them back?\n\nClick YES to end the innings.\nClick NO to recall a retired batter."
                ):
                    self.end_innings()
                    return
                else:
                    self.select_returning_batter(possibly_returnable)
                    return
            else:
                self.end_innings()
                return

        # Select new batter
        selector = ttkb.Toplevel(self.window)
        selector.title("Select New Batter")
        selector.grab_set()

        # ❌ Prevent closing the window without choosing
        selector.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning(
            "Required", "You must select a new batter to continue."))

        self.new_batter = tk.IntVar()

        for pid, name in available:
            ttkb.Radiobutton(selector, text=name, variable=self.new_batter,
                            value=pid).pack(anchor=tk.W)

        ttkb.Button(selector, text="Confirm",
                    command=lambda: self.finalize_batter_change(selector, surviving_batter),
                    bootstyle=SUCCESS).pack(pady=10)

    def select_returning_batter(self, returnable_ids):
        selector = tk.Toplevel(self.window)
        selector.title("Recall Retired Not Out Batter")
        selector.geometry("400x400")
        selector.grab_set()
        self.new_batter = tk.IntVar()

        for pid in returnable_ids:
            name = self.get_player_name(pid)
            ttk.Radiobutton(selector, text=name, variable=self.new_batter, value=pid).pack(anchor=tk.W)

        ttk.Button(selector, text="Confirm", command=lambda: self.finalize_batter_change(selector, self.match_data.striker)).pack(pady=10)

    def finalize_batter_change(self, window, surviving_batter):
        window.destroy()
        
        # Get selected player ID directly from the radio button value
        new_batter_id = self.new_batter.get()

        # Update match data directly with the ID
        self.match_data.new_batter = new_batter_id
        
        # Update striker/non-striker based on who was dismissed
        if self.match_data.dismissed_batter == self.match_data.non_striker:
            self.match_data.striker = surviving_batter
            self.match_data.non_striker = new_batter_id
            self.match_data.current_partnership['batter1'] = surviving_batter
            self.match_data.current_partnership['batter2'] = new_batter_id
        else:
            self.match_data.non_striker = surviving_batter
            self.match_data.striker = new_batter_id
            self.match_data.current_partnership['batter1'] = new_batter_id
            self.match_data.current_partnership['batter2'] = surviving_batter

        # Initialize new batter stats
        self.match_data.batters[new_batter_id] = {
            'runs': 0,
            'balls': 0,
            'fours': 0,
            'sixes': 0,
            'status': 'not out'
        }


        # Start new partnership using current striker/non-striker
        ball_data = {'dismissed_player_id': self.match_data.dismissed_batter}
        self._start_new_partnership(ball_data)

        self.update_display()      

    def select_new_bowler(self):
        """Show bowler selection with proper stats handling"""
        selector = ttkb.Toplevel(self.window)
        selector.title("Select Next Bowler")
        selector.geometry("400x400")
        selector.grab_set()

        # 🔒 Prevent closing the window without selection
        selector.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning(
            "Required", "You must select a bowler to continue."))

        # Get valid bowlers from the playing XI
        bowling_team_players = [
            p for p in self.match_data.selected_players[self.match_data.bowling_team]
            if p not in (self.match_data.team1_twelfth + self.match_data.team2_twelfth)
        ]

        # Ensure all potential bowlers are initialized
        for pid in bowling_team_players:
            if pid not in self.match_data.bowlers:
                self.match_data.bowlers[pid] = {
                    'balls': 0, 'runs': 0, 'wickets': 0,
                    'maidens': 0, 'dot_balls': 0,
                    'wides': 0, 'no_balls': 0
                }

        # Create bowler list with safe stats access
        bowler_info = []
        for pid in bowling_team_players:
            try:
                name = self.get_player_name(pid)
                balls = self.match_data.bowlers[pid]['balls']
                overs = f"{balls//6}.{balls%6}"
                bowler_info.append((pid, name, overs))
            except KeyError:
                continue

        if not bowler_info:
            messagebox.showerror("Error", "No available bowlers")
            return

        self.new_bowler = tk.IntVar(value=bowler_info[0][0])

        for pid, name, overs in bowler_info:
            btn_text = f"{name} (Overs: {overs})"
            ttkb.Radiobutton(selector, text=btn_text, variable=self.new_bowler,
                            value=pid).pack(anchor=tk.W)

        ttkb.Button(selector, text="Confirm",
                    command=lambda: self.finalize_bowler_selection(selector)).pack(pady=10)
    
    def update_bowler(self, window):
        window.destroy()
        new_bowler = self.new_bowler.get()
        self.match_data.current_bowler = new_bowler
        
        # Ensure bowler exists in stats (for non-defaultdict case)
        if new_bowler not in self.match_data.bowlers:
            self.match_data.bowlers[new_bowler] = {
                'balls': 0, 'runs': 0, 'wickets': 0,
                'maidens': 0, 'dot_balls': 0,
                'wides': 0, 'no_balls': 0
            }
        
        self.update_display()

    def finalize_bowler_selection(self, window):
        window.destroy()
        new_bowler = self.new_bowler.get()
        
        # Initialize if not exists (defensive check)
        if new_bowler not in self.match_data.bowlers:
            self.match_data.bowlers[new_bowler] = {
                'balls': 0, 'runs': 0, 'wickets': 0,
                'maidens': 0, 'dot_balls': 0,
                'wides': 0, 'no_balls': 0
            }
        
        self.match_data.current_bowler = new_bowler
        self.update_display()

    def _finalize_partnership(self, unbeaten=False, previous_partnership=None):
        conn = None
        try:
            p = previous_partnership or self.match_data.current_partnership  # fallback if needed

            b1, b2 = sorted([p['batter1'], p['batter2']])
            end_over = round(self.match_data.current_over + (self.match_data.balls_this_over / 6), 1)

            #print("📌 FINALIZING PARTNERSHIP")
            #print(f"   ➤ Wickets: {self.match_data.wickets}")
            #print(f"   ➤ Current Over: {self.match_data.current_over}")
            #print(f"   ➤ Balls This Over: {self.match_data.balls_this_over}")
            #print(f"   ➤ Computed End Over: {end_over}")
            #print(f"   ➤ Batters (sorted): {b1} & {b2}")

            conn = sqlite3.connect("C:/Users/Danielle/Desktop/Cricket Analysis Program/cricket_analysis.db")
            c = conn.cursor()

            c.execute('''
                UPDATE partnerships
                SET 
                    runs = ?,
                    balls = ?,
                    dots = ?,
                    ones = ?,
                    twos = ?,
                    threes = ?,
                    fours = ?,
                    sixes = ?,
                    end_over = ?,
                    unbeaten = ?
                WHERE innings_id = ? AND batter1_id = ? AND batter2_id = ?
            ''', (
                p['runs'], p['balls'], p['dots'], p['ones'], p['twos'],
                p['threes'], p['fours'], p['sixes'],
                end_over,
                1 if unbeaten else 0,
                self.match_data.innings_id,
                b1, b2
            ))

            #print(f"🔍 Rows affected by update: {c.rowcount}")
            conn.commit()
            #print(f"✅ Finalized partnership between {b1} and {b2}")

        except Exception as e:
            print(f"❌ Error finalizing partnership: {e}")
            import traceback
            traceback.print_exc()
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
                #print("🔒 Database connection closed")

    def _start_new_partnership(self, ball_data=None, opening=False):
        try:
            # 🔐 Validate core info
            if not self.match_data.striker or not self.match_data.non_striker:
                raise ValueError("❌ Striker or non-striker not set properly.")
            if not self.match_data.bowling_team:
                raise ValueError("❌ Bowling team not set.")
            if not hasattr(self.match_data, 'wickets'):
                self.match_data.wickets = 0

            # 🎯 Pull batters from current match data
            batter1 = self.match_data.striker
            batter2 = self.match_data.non_striker

            # 🔒 Safety check: can't be the same player
            if batter1 == batter2:
                raise ValueError("🛑 Surviving batter and new batter are the same!")

            # ✅ Sort for consistent DB order
            batter1_id, batter2_id = sorted([batter1, batter2])

            # 📌 Define start wicket and over
            start_wicket = 1 if opening else max(1, self.match_data.wickets + 1)
            start_over = self.match_data.current_over + (self.match_data.balls_this_over / 6)
            opponent_team = self.match_data.bowling_team

            #print(f"📌 Starting new partnership - Wicket: {start_wicket}, Opponent: {opponent_team}")
            #print(f"👥 Batters: {batter1} & {batter2} (Sorted: {batter1_id}, {batter2_id})")

            # Store in match_data
            self.match_data.current_partnership = {
                'batter1': batter1_id,
                'batter2': batter2_id,
                'runs': 0,
                'balls': 0,
                'dots': 0,
                'ones': 0,
                'twos': 0,
                'threes': 0,
                'fours': 0,
                'sixes': 0,
                'last_milestone': 0,
                'start_wicket': start_wicket,
                'start_over': start_over,
                'unbeaten': 0
            }

            # ➕ Insert into DB
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            c.execute('''INSERT INTO partnerships (
                innings_id, start_wicket, batter1_id, batter2_id, runs, balls,
                dots, ones, twos, threes, fours, sixes, start_over, opponent_team, unbeaten
            ) VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, ?, ?, 0)''', (
                self.match_data.innings_id,
                start_wicket,
                batter1_id,
                batter2_id,
                start_over,
                opponent_team
            ))
            conn.commit()
            #print(f"✅ Partnership inserted successfully for batters {batter1_id} & {batter2_id}")

        except Exception as e:
            print(f"❌ Failed to start new partnership: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in locals():
                conn.rollback()
        finally:
            if 'conn' in locals():
                conn.close()

    def get_ball_symbol(self, ball_data):
        """Convert ball event data to display symbol"""
        # Check for dismissal first
        if ball_data.get('dismissal'):
            return 'W'
        
        extras = ball_data.get('extras', {})
        
        # Check extras in priority order
        if extras.get('wides', 0) > 0:
            return f"Wd{extras['wides']}"
        if extras.get('no_balls', 0) > 0:
            batters_runs = ball_data.get('runs', 0)
            penalty_runs = ball_data.get('penalty_runs', 0)
            if batters_runs != 0 and penalty_runs == 0:
                return f"Nb+{batters_runs}"
            bye_runs = ball_data.get('byes', 0)
            if bye_runs != 0:
                return f"Nb+{bye_runs}b"
            leg_bye_runs = ball_data.get('leg_byes', 0)
            if leg_bye_runs != 0:
                return f"Nb+{leg_bye_runs}lb"
            if penalty_runs != 0 and batters_runs == 0 and bye_runs == 0 and leg_bye_runs == 0:
                return f"Nb+{penalty_runs}P"
            if penalty_runs != 0 and batters_runs != 0:
                return f"Nb+{penalty_runs}P+{batters_runs}"
            if penalty_runs != 0 and bye_runs != 0:
                return f"Nb+{penalty_runs}P+{bye_runs}b"
            if penalty_runs != 0 and leg_bye_runs != 0:
                return f"Nb+{penalty_runs}P+{leg_bye_runs}lb"
            if batters_runs == 0 and bye_runs == 0 and leg_bye_runs == 0 and penalty_runs == 0:
                return f"Nb"
        if extras.get('byes', 0) > 0:
            return f"B{extras['byes']}"
        if extras.get('leg_byes', 0) > 0:
            return f"Lb{extras['leg_byes']}"
        if extras.get('penalty', 0) > 0:
            return f"P{extras['penalty']}"
        
        # Regular ball
        return str(ball_data.get('runs', 0))

    def update_display(self, reset=False):
            # ✅ Reset display panel completely (used at new innings start)
        if reset:
            self.score_label.config(text="0/0")
            self.overs_label.config(text="Overs: 0.0")
            self.striker_label.config(text="Striker: -")
            self.non_striker_label.config(text="Non-Striker: -")
            self.bowler_label.config(text="Bowler: -")
            self.over_display.config(text="-")
            self.current_rr_var.set("0.00")
            self.required_rr_var.set("N/A")
            self.target_label.config(text="")
            return

        # Show adjusted target if applicable
        if self.match_data.innings == 2 and self.match_data.target_runs:
            if self.match_data.was_rain_delayed:
                self.target_label.config(text=f"Target: {self.match_data.target_runs} (Rain Adj.)")
            else:
                self.target_label.config(text=f"Target: {self.match_data.target_runs}")
        else:
            self.target_label.config(text="")


        total_balls = (self.match_data.current_over * 6) + self.match_data.balls_this_over
        if total_balls > 0:
            current_rr = (self.match_data.total_runs / total_balls) * 6
            self.current_rr_var.set(f"{current_rr:.2f}")
        else:
            self.current_rr_var.set("0.00")

        # Calculate Required Run Rate (RRR) — only in 2nd innings
        if self.match_data.innings == 2 and self.match_data.target_runs is not None:
            remaining_runs = self.match_data.target_runs - self.match_data.total_runs
            remaining_balls = ((self.match_data.total_overs * 6) -
                            (self.match_data.current_over * 6 + self.match_data.balls_this_over))
            if remaining_balls > 0 and remaining_runs > 0:
                required_rr = (remaining_runs / remaining_balls) * 6
                self.required_rr_var.set(f"{required_rr:.2f}")
            else:
                self.required_rr_var.set("N/A")
        else:
            self.required_rr_var.set("N/A")

        # Score and Overs
        self.score_label.config(text=f"{self.match_data.total_runs}/{self.match_data.wickets}")
        overs_text = f"Overs: {self.match_data.current_over:.0f}.{self.match_data.balls_this_over}"
        if self.match_data.was_rain_delayed:
            overs_text += f" / {self.match_data.total_overs} (Adj.)"
        self.overs_label.config(text=overs_text)

        # Determine current phase using overs_phases
        total_balls_bowled = self.match_data.current_over * 6 + self.match_data.balls_this_over
        current_over_float = round(total_balls_bowled / 6, 2)
        #print("[DEBUG] Overs Phases:", self.match_data.overs_phases)
        #print("[DEBUG] Current Over Float:", current_over_float)
        phase_text = "-"
        for phase, (start, end) in self.match_data.overs_phases.items():
            if start <= current_over_float <= end:
                phase_text = phase
                break

        self.phase_label.config(text=f"Phase: {phase_text}")
        
        # Batters
        striker_stats = self.match_data.batters.get(self.match_data.striker, {
            'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'
        })
        self.striker_label.config(text=f"{self.get_player_name(self.match_data.striker)}: {striker_stats['runs']} ({striker_stats['balls']}) ★")
        non_striker_stats = self.match_data.batters.get(self.match_data.non_striker, {
            'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'status': 'not out'
        })
        self.non_striker_label.config(text=f"{self.get_player_name(self.match_data.non_striker)}: {non_striker_stats['runs']} ({non_striker_stats['balls']})")
        
        # Bowler
        bowler_stats = self.match_data.bowlers.get(self.match_data.current_bowler, {
            'balls': 0, 'runs': 0, 'wickets': 0, 'maidens': 0, 'dot_balls': 0, 'wides': 0, 'no_balls': 0
        })
        overs = f"{bowler_stats['balls'] // 6}.{bowler_stats['balls'] % 6}"
        self.bowler_label.config(text=f"{self.get_player_name(self.match_data.current_bowler)}: {overs} - {bowler_stats['runs']}/{bowler_stats['wickets']}")
        
        # Update last over display using symbols
        current_balls = [b['symbol'] for b in self.match_data.current_over_balls[-6:]]  # Extract symbols
        ball_symbols = []
        for symbol in current_balls:
            if symbol.startswith('Nb'):
                ball_symbols.append(f"[{symbol}]")
            else:
                ball_symbols.append(symbol)
        self.over_display.config(text=" ".join(ball_symbols))

        # Update tables
        self.update_stats_trees()

    def update_stats_trees(self):
        # Update batters tree
        self.batters_tree.delete(*self.batters_tree.get_children())
        for player_id, stats in self.match_data.batters.items():
            name = self.get_player_name(player_id)
            sr = stats['runs'] / stats['balls'] * 100 if stats['balls'] > 0 else 0
            self.batters_tree.insert('', 'end', values=(
                name,
                stats['runs'],
                stats['balls'],
                stats['fours'],
                stats['sixes'],
                f"{sr:.1f}",
                stats.get('status', 'not out')
            ))

        # Populate bowlers with valid names
        for player_id, stats in self.match_data.bowlers.items():
            name = self.get_player_name(player_id)
            if name == "Unknown Player":
                continue  # Skip invalid entries

        # Clear existing data
        self.bowlers_tree.delete(*self.bowlers_tree.get_children())
        
        # Only show bowlers who have actually bowled
        for player_id in list(self.match_data.bowlers.keys()):
            # Skip bowlers with zero balls and invalid IDs
            if not isinstance(player_id, int):
                continue
                
            name = self.get_player_name(player_id)
            stats = self.match_data.bowlers[player_id]
            
            # Calculate display values
            overs = f"{stats['balls'] // 6}.{stats['balls'] % 6}"
            economy = stats['runs']/(stats['balls']/6) if stats['balls'] > 0 else 0
            
            # Insert into treeview
            self.bowlers_tree.insert('', 'end', values=(
                name,
                overs,
                stats['dot_balls'],
                stats['runs'],
                stats['wickets'],
                f"{economy:.2f}",
                stats['wides'],
                stats['no_balls']
            ))

    def get_player_name(self, player_id):
        """Safely get player name with validation"""
        if not isinstance(player_id, int):
            return "Invalid Player ID"
            
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        try:
            c = conn.cursor()
            c.execute("SELECT player_name FROM players WHERE player_id = ?", (player_id,))
            result = c.fetchone()
            return result[0] if result else "Unknown Player"
        except sqlite3.Error as e:
            print(f"Database Error: {str(e)}")
            return "Error"
        finally:
            conn.close()

    def get_fielder_id(self):
        fielder_name = self.fielder_combo.get()
        if not fielder_name:
            return None
            
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute("SELECT player_id FROM players WHERE player_name = ?", (fielder_name,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

    def clear_inputs(self):
        # Reset all input fields
        self.runs_var.set(0)
        self.dismissal_var.set(False)
        self.dismissal_combo.set('')
        self.shot_type_var.set('Rotation')
        self.shot_selection_combo.set('')
        self.footwork_var.set('Nothing')
        self.aerial_var.set(False)
        self.edged_var.set(False)
        self.missed_var.set(False)
        self.clean_hit_var.set(False)
        self.delivery_combo.set('')
        self.fielding_style_combo.set('')
        self.expected_runs_var.set(0)
        self.fielder_combo.set('')
        self.expected_manually_changed = False
        self.batter_blind_turn_var.set(False)
        self.non_striker_blind_turn_var.set(False)

        
        # Reset fielding checkboxes
        for var in self.fielding_vars.values():
            var.set(False)

        # Clear extras
        for var in self.extras_vars.values():
            var.set(0)
        
        # Clear fielding event checkboxes
        for var in self.fielding_events.values():
            var.set(False)
            
        # Clear canvas markers
        self.pitch_canvas.delete("pitch_marker")
        self.wagon_canvas.delete("shot_line")

    def save_ball_event(self, ball_data):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        try:
            if not hasattr(self.match_data, 'innings_id') or not self.match_data.innings_id:
                raise ValueError("Missing innings_id in match_data")

            if not hasattr(self.match_data, "match_id") or not self.match_data.match_id:
                messagebox.showerror("Error", "No active match! Create a match first.")
                return 0

            # === Free Hit Logic ===
            ball_data['free_hit'] = 0
            c.execute("""
                SELECT no_balls FROM ball_events
                WHERE innings_id = ?
                ORDER BY ball_id DESC
                LIMIT 1
            """, (self.match_data.innings_id,))
            prev = c.fetchone()
            if prev and prev[0] > 0:
                ball_data['free_hit'] = 1

            # === Batting Position Logic ===
            batting_team = self.match_data.batting_team
            striker_id = self.match_data.striker
            batting_order_list = self.match_data.selected_players.get(batting_team, [])
            if striker_id in batting_order_list:
                ball_data["batting_position"] = batting_order_list.index(striker_id) + 1

            # === Bowling Order Logic ===
            bowling_team = self.match_data.bowling_team
            bowler_id = self.match_data.current_bowler
            if not hasattr(self.match_data, "bowler_usage"):
                self.match_data.bowler_usage = {}
            bowling_order_list = self.match_data.bowler_usage.setdefault(bowling_team, [])
            if bowler_id not in bowling_order_list:
                bowling_order_list.append(bowler_id)
            ball_data["bowling_order"] = bowling_order_list.index(bowler_id) + 1

            # === Shot Coordinates ===
            shot_x, shot_y = None, None
            if ball_data.get('shot_coords'):
                try:
                    shot_x = ball_data['shot_coords']['cartesian'][0]
                    shot_y = ball_data['shot_coords']['cartesian'][1]
                except (KeyError, TypeError):
                    pass

            # === Fielding Style ===
            fielding_style = self.fielding_style_combo.get() or None

            # === Final Data Prep ===
            ball_data['fielder'] = self.get_fielder_id()
            ball_data['fielding_events'] = ball_data.get('fielding_events', [])

            # === Dynamic Game Phase Tagging ===
            # Note: Use 1-based over for phase logic (0.3 → 1st over)
            current_over_display = self.match_data.current_over + 1
            batter_blind_turn = int(self.batter_blind_turn_var.get())
            non_striker_blind_turn = int(self.non_striker_blind_turn_var.get()) 
            phases = self.match_data.overs_phases
            pp_start, pp_end = phases['Powerplay']
            mo_start, mo_end = phases['Middle Overs']
            do_start, do_end = phases['Death Overs']

            ball_data['is_powerplay'] = int(pp_start <= current_over_display <= pp_end)
            ball_data['is_middle_overs'] = int(mo_start <= current_over_display <= mo_end)
            ball_data['is_death_overs'] = int(do_start <= current_over_display <= do_end)

            over_number = self.match_data.current_over
            ball_number = self.match_data.balls_this_over

            if ball_number == 0:
                ball_number = 6
                over_number -= 1

            #print(f"[DEBUG] Saving expected_wicket: {ball_data.get('expected_wicket')}")
            print(f"💾 Final Intent Score to DB: {ball_data['intent_score']}")
            params = (
                self.match_data.innings_id,
                over_number,
                self.match_data.innings,
                ball_number,
                int(ball_data.get('ball_number', 0)),
                self.match_data.striker,
                self.match_data.non_striker,
                self.match_data.current_bowler,
                ball_data.get('fielder'),
                int(ball_data.get('runs', 0)),
                json.dumps(ball_data.get('extras', {})) if ball_data.get('extras') else None,
                self.shot_type_var.get(),
                self.footwork_var.get(),
                int(self.aerial_var.get()),
                self.dismissal_combo.get() if self.dismissal_var.get() else None,
                ball_data.get('dismissed_player_id'),
                self.current_pitch_location[0] if self.current_pitch_location else None,
                self.current_pitch_location[1] if self.current_pitch_location else None,
                shot_x,
                shot_y,
                ball_data.get('delivery_type', ''),
                fielding_style,
                int(ball_data.get('edged', 0)),
                ball_data.get('shot_selection', ''),
                int(ball_data.get('dot_balls', 0)),
                int(ball_data.get('wides', 0)),
                int(ball_data.get('no_balls', 0)),
                int(ball_data.get('free_hit', 0)),
                int(ball_data.get('byes', 0)),
                int(ball_data.get('leg_byes', 0)),
                int(ball_data.get('penalty_runs', 0)),
                int(ball_data.get('missed', 0)),
                int(ball_data.get('clean_hit')),
                int(ball_data.get('expected_runs', 0)),
                float(ball_data.get('expected_wicket', 0.00)),
                round(float(ball_data.get('batting_bpi', 0)), 2),
                round(float(ball_data.get('bowling_bpi', 0)), 2),
                round(float(ball_data.get('intent_score', 0)), 2),
                int(ball_data.get('batting_position', 0)),
                int(ball_data.get('bowling_order', 0)),
                batter_blind_turn,
                non_striker_blind_turn,
                int(ball_data.get('over_the_wicket', 0)),
                int(ball_data.get('around_the_wicket', 0)),
                int(ball_data.get('is_powerplay', 0)),
                int(ball_data.get('is_middle_overs', 0)),
                int(ball_data.get('is_death_overs', 0))
            )

            # === Insert Ball Event ===
            c.execute('''INSERT INTO ball_events (
                innings_id, over_number, innings, balls_this_over, ball_number,
                batter_id, non_striker_id, bowler_id, fielder_id, runs, extras,
                shot_type, footwork, aerial, dismissal_type,
                dismissed_player_id, pitch_x, pitch_y,
                shot_x, shot_y, delivery_type, fielding_style,
                edged, shot_selection, dot_balls, wides,
                no_balls, free_hit, byes, leg_byes, penalty_runs,
                ball_missed, clean_hit, expected_runs, expected_wicket, batting_bpi, bowling_bpi, batting_intent_score,
                batting_position, bowling_order, batter_blind_turn, non_striker_blind_turn, over_the_wicket, around_the_wicket, is_powerplay, is_middle_overs, is_death_overs
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', params)

            ball_id = c.lastrowid

            # === Update or Insert Partnership Info ===
            # Normalize batters regardless of striker/non-striker swap
            b1 = self.match_data.striker
            b2 = self.match_data.non_striker
            batter1_id, batter2_id = sorted([b1, b2])

            # Check for existing partnership
            c.execute("""
                SELECT partnership_id FROM partnerships
                WHERE innings_id = ? AND batter1_id = ? AND batter2_id = ?
                ORDER BY partnership_id DESC LIMIT 1
            """, (self.match_data.innings_id, batter1_id, batter2_id))

            row = c.fetchone()

            if row:
                partnership_id = row[0]
                c.execute("""
                    UPDATE partnerships
                    SET runs = runs + ?, balls = balls + 1
                    WHERE partnership_id = ?
                """, (ball_data.get('runs', 0), partnership_id))
            else:
                print("⚠️ Warning: No current partnership found! Not inserting automatically to avoid duplicates.")
                # Optional: raise an error or log only — don't insert blindly


            # === Insert Fielding Events ===
            for event_name in ball_data['fielding_events']:
                c.execute("SELECT event_id FROM fielding_events WHERE event_name = ?", (event_name,))
                event_id = c.fetchone()
                if event_id:
                    c.execute("INSERT INTO ball_fielding_events (ball_id, event_id) VALUES (?, ?)", (ball_id, event_id[0]))

            # === Insert Fielder Contribution ===
            if ball_data['fielder']:
                c.execute('''INSERT INTO fielding_contributions
                            (ball_id, fielder_id, boundary_saved)
                            VALUES (?, ?, ?)''',
                        (ball_id, ball_data['fielder'], ball_data.get('boundary_saved', 0)))

            # === Update Innings Table ===
            c.execute("""
                UPDATE innings
                SET 
                    total_runs = (
                        SELECT COALESCE(SUM(runs)
                            + SUM(CAST(json_extract(extras, '$.wides') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.no_balls') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.byes') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.leg_byes') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.penalty') AS INTEGER)), 0)
                        FROM ball_events
                        WHERE innings_id = ?
                    ),
                    wickets = (
                        SELECT COUNT(*) FROM ball_events
                        WHERE innings_id = ?
                        AND dismissal_type IS NOT NULL
                        AND dismissal_type != ''
                    ),
                    overs_bowled = (
                        SELECT CAST(COUNT(*) FILTER (
                            WHERE b.innings_id = ? AND b.wides = 0 AND b.no_balls = 0
                        ) AS FLOAT) / 6.0
                        FROM ball_events b
                    ),
                    extras = (
                        SELECT COALESCE(
                            SUM(CAST(json_extract(extras, '$.wides') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.no_balls') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.byes') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.leg_byes') AS INTEGER))
                            + SUM(CAST(json_extract(extras, '$.penalty') AS INTEGER)), 0)
                        FROM ball_events
                        WHERE innings_id = ?
                    )
                WHERE innings_id = ?
            """, (
                self.match_data.innings_id,
                self.match_data.innings_id,
                self.match_data.innings_id,
                self.match_data.innings_id,
                self.match_data.innings_id
            ))




            # ✅ Only one commit here
            conn.commit()
            #print(f"[SAVE COMPLETE] Ball ID {ball_id} committed successfully.")
            return ball_id

        except sqlite3.Error as e:
            conn.rollback()
            print("[DB ERROR]", str(e))
            import traceback
            traceback.print_exc()
            messagebox.showerror("Save Error", f"Database Error:\n{str(e)}")
            return 0

        finally:
            conn.close()

    def save_state(self):
        """Deep copy all relevant data for proper undo"""
        import copy
        self.history.append({
            'total_runs': self.match_data.total_runs,
            'wickets': self.match_data.wickets,
            'current_over': self.match_data.current_over,
            'balls_this_over': self.match_data.balls_this_over,
            'striker': self.match_data.striker,
            'non_striker': self.match_data.non_striker,
            'current_bowler': self.match_data.current_bowler,
            'batters': copy.deepcopy(self.match_data.batters),
            'bowlers': copy.deepcopy(dict(self.match_data.bowlers)),  # Convert defaultdict
            'dismissed_players': copy.deepcopy(self.match_data.dismissed_players),
            'current_over_balls': copy.deepcopy(self.match_data.current_over_balls),
            'batting_bpi': self.match_data.batting_bpi,
            'bowling_bpi': self.match_data.bowling_bpi,
            'current_partnership': copy.deepcopy(self.match_data.current_partnership)
        })

    def save_match_state_to_db(self):
        state = {
            # your existing saved keys...
            "total_runs": self.match_data.total_runs,
            "wickets": self.match_data.wickets,
            "current_over": self.match_data.current_over,
            "balls_this_over": self.match_data.balls_this_over,
            "striker": self.match_data.striker,
            "non_striker": self.match_data.non_striker,
            "current_bowler": self.match_data.current_bowler,
            "batters": self.match_data.batters,
            "bowlers": self.match_data.bowlers,
            "current_partnership": self.match_data.current_partnership,
            "dismissed_players": list(self.match_data.dismissed_players),
            "retired_not_out_players": list(self.match_data.retired_not_out_players),
            "batting_bpi": self.match_data.batting_bpi,
            "bowling_bpi": self.match_data.bowling_bpi,
            "current_over_balls": self.match_data.current_over_balls,
            "waiting_for_new_innings_setup": self.match_data.waiting_for_new_innings_setup,
            # Add these lines explicitly
            "batting_team": self.match_data.batting_team,
            "bowling_team": self.match_data.bowling_team,
            "selected_players": self.match_data.selected_players
        }

        json_state = json.dumps(state)

        conn = sqlite3.connect("C:/Users/Danielle/Desktop/Cricket Analysis Program/cricket_analysis.db")
        c = conn.cursor()
        c.execute("UPDATE innings SET saved_state = ? WHERE innings_id = ?", (json_state, self.match_data.innings_id))
        conn.commit()
        conn.close()

    def undo_ball(self):
        if not self.history:
            messagebox.showinfo("Undo", "No ball to undo.")
            return

        confirm = messagebox.askyesno("Confirm Undo", "Are you sure you want to undo the last ball?")
        if not confirm:
            return
        

        state = self.history.pop()
        # Restore all values
        self.match_data.total_runs = state['total_runs']
        self.match_data.wickets = state['wickets']
        self.match_data.current_over = state['current_over']
        self.match_data.balls_this_over = state['balls_this_over']
        self.match_data.striker = state['striker']
        self.match_data.non_striker = state['non_striker']
        self.match_data.current_bowler = state['current_bowler']
        self.match_data.batters = state['batters']
        self.match_data.bowlers = defaultdict(
            lambda: {'balls':0, 'runs':0, 'wickets':0, 'maidens':0, 'dot_balls':0, 'wides':0, 'no_balls':0},
            state['bowlers']
        )
        self.match_data.dismissed_players = state['dismissed_players']
        self.match_data.current_over_balls = state['current_over_balls']
        self.match_data.batting_bpi = state['batting_bpi']
        self.match_data.bowling_bpi = state['bowling_bpi']
        if self.match_data.wickets < state['wickets']:
            self.delete_last_partnership()
        self.match_data.current_partnership = state.get('current_partnership', None)
        self.delete_last_ball_from_db()
        self.update_display()
        self.update_stats_trees() 

    def swap_batters(self):
        self.match_data.striker, self.match_data.non_striker = (
            self.match_data.non_striker, self.match_data.striker
        )
        self.update_blind_turn_labels()
        self.update_display()

    def delete_last_partnership(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()
        c.execute('''
            DELETE FROM partnerships 
            WHERE innings_id = ? 
            ORDER BY start_over DESC 
            LIMIT 1
        ''', (self.match_data.innings_id,))
        conn.commit()
        conn.close()

    def delete_last_ball_from_db(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Get the last ball_id for the current innings
        c.execute("""
            SELECT ball_id FROM ball_events 
            WHERE innings_id = ? 
            ORDER BY ball_id DESC 
            LIMIT 1
        """, (self.match_data.innings_id,))
        last_ball = c.fetchone()

        if last_ball:
            ball_id = last_ball[0]
            
            # Delete associated fielding events (if any)
            c.execute("DELETE FROM ball_fielding_events WHERE ball_id = ?", (ball_id,))

            # Delete shot placement or other linked tables if applicable
            # c.execute("DELETE FROM ball_shot_placement WHERE ball_id = ?", (ball_id,))
            c.execute("DELETE FROM player_pressure_impact WHERE ball_id = ?", (ball_id,))
            # Delete the ball event itself
            c.execute("DELETE FROM ball_events WHERE ball_id = ?", (ball_id,))

            conn.commit()

        conn.close()

    def end_innings(self):
        try:
            self.match_data.innings_ended = True

            # Finalize current partnership if it has any balls
            if self.match_data.current_partnership and self.match_data.current_partnership['balls'] > 0:
                self._finalize_partnership(unbeaten=True)

            # Mark innings as completed in the DB
            conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
            c = conn.cursor()
            c.execute("UPDATE innings SET completed = 1 WHERE innings_id = ?", (self.match_data.innings_id,))
            conn.commit()
            conn.close()

            # Display summary
            messagebox.showinfo(
                "Innings Over",
                f"Innings concluded at {self.match_data.total_runs}/{self.match_data.wickets}\n"
                f"Overs: {self.match_data.current_over:.0f}.{self.match_data.balls_this_over}"
            )
            #print(f"✅ Innings {self.match_data.innings_id} marked as completed.")

            # Disable input temporarily
            self.submit_btn.configure(state=tk.DISABLED)

            if self.match_data.innings == 2:
                runs = self.match_data.total_runs
                target = self.match_data.target_runs

                if runs == target - 1:
                    self.trigger_super_over()
                else:
                    winner = self.match_data.bowling_team if runs < target - 1 else self.match_data.batting_team
                    self.end_match(winner=winner)

            elif self.match_data.innings == 1:
                self.start_second_innings()  # ✅ Only now call this

            self.save_match_state_to_db()

        except Exception as e:
            print(f"❌ Error ending innings: {e}")
            messagebox.showerror("Error", f"Could not finalize the innings:\n{e}")

    def _confirm_end_innings(self):
        result = messagebox.askyesno("Confirm End Innings", "Are you sure you want to end the innings?")
        if result:
            self.end_innings()

    def start_second_innings(self):
        #print("🚨 Starting second innings")
        # Set target based on first innings total
        target = self.match_data.total_runs
        self.match_data.innings = 2
        self.match_data.target_runs = target + 1

        # 🔁 Swap teams BEFORE inserting new innings
        self.match_data.batting_team, self.match_data.bowling_team = \
            self.match_data.bowling_team, self.match_data.batting_team

        # 🆕 Insert new innings into DB
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        c.execute('''
            INSERT INTO innings (match_id, innings, batting_team, bowling_team, completed)
            VALUES (?, ?, ?, ?, 0)
        ''', (
            self.match_data.match_id,
            2,
            self.match_data.batting_team,
            self.match_data.bowling_team
        ))

        conn.commit()
        self.match_data.innings_id = c.lastrowid
        conn.close()

        # 🔁 Reset match state
        self.match_data.total_runs = 0
        self.match_data.wickets = 0
        self.match_data.current_over = 0
        self.match_data.balls_this_over = 0
        self.match_data.batting_bpi = 0
        self.match_data.bowling_bpi = 0
        self.match_data.striker = None
        self.match_data.non_striker = None
        self.match_data.new_batter = None
        self.match_data.batters = {}
        self.match_data.bowlers = {}
        self.match_data.dismissed_players = set()
        self.match_data.current_partnership = None
        self.match_data.current_over_balls = []

        # ✅ Reset UI
        self.update_display(reset=True)
        self.clear_stats_trees()
        self.set_submit_enabled(True)
        self.update_bpi_display(0.0, 0.0)

        # 🔄 Update fielding options
        self.update_fielder_options()

        if self.match_data.was_rain_delayed:
            self.prompt_adjust_target()

        # ✅ Prompt new inputs
        self.prompt_opening_batters()
        self.select_opening_bowler()

        self.match_data.waiting_for_new_innings_setup = True
        print(f"🟨 waiting_for_new_innings_setup set to True — {self.match_data.waiting_for_new_innings_setup}")

    def get_previous_ball_events(self):
        conn = sqlite3.connect(r"C:\Users\Danielle\Desktop\Cricket Analysis Program\cricket_analysis.db")
        c = conn.cursor()

        # Step 1: Fetch main ball event data
        c.execute('''
            SELECT 
                ball_id, innings_id, over_number, innings, balls_this_over, ball_number, batter_id,
                bowler_id, fielder_id, runs, extras, shot_type, footwork,
                shot_selection, aerial, dismissal_type, dismissed_player_id,
                pitch_x, pitch_y, shot_x, shot_y, delivery_type, fielding_style,
                edged, ball_missed, wides, no_balls, byes, leg_byes,
                penalty_runs, dot_balls, expected_runs, is_powerplay,
                is_middle_overs, is_death_overs
            FROM ball_events
            WHERE innings_id = ?
        ''', (self.match_data.innings_id,))
        
        columns = [col[0] for col in c.description]
        balls = [dict(zip(columns, row)) for row in c.fetchall()]

        # Step 2: Fetch and assign fielding events for each ball
        for ball in balls:
            c.execute('''
                SELECT fe.event_name
                FROM ball_fielding_events bfe
                JOIN fielding_events fe ON bfe.event_id = fe.event_id
                WHERE bfe.ball_id = ?
            ''', (ball["ball_id"],))
            ball["fielding_events"] = [row[0] for row in c.fetchall()]

        conn.close()
        return balls

    def prompt_opening_batters(self):
        window = ttkb.Toplevel(self.window)
        window.title("Select Opening Batters")
        window.geometry("400x400")
        window.grab_set()

        
        # 🚫 Prevent accidental closing
        window.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning(
            "Required", "You must select two opening batters to continue."))

        ttkb.Label(window, text="Select two opening batters:").pack(pady=10)

        # Fetch playing XI for the new batting team
        team = self.match_data.batting_team
        playing_xi = self.match_data.selected_players[team]

        # Exclude any twelfth players (optional)
        twelfth = self.match_data.team1_twelfth + self.match_data.team2_twelfth
        eligible_batters = [pid for pid in playing_xi if pid not in twelfth]

        # Display as checkboxes
        selected_batters = []

        def toggle(pid):
            if pid in selected_batters:
                selected_batters.remove(pid)
            else:
                if len(selected_batters) < 2:
                    selected_batters.append(pid)
                else:
                    messagebox.showwarning("Too many selected", "Please select only 2 opening batters.")

        for pid in eligible_batters:
            name = self.get_player_name(pid)
            cb = ttkb.Checkbutton(window, text=name, command=lambda p=pid: toggle(p))
            cb.pack(anchor=tk.W)

        def confirm_selection():
            if len(selected_batters) != 2:
                messagebox.showerror("Selection Error", "Please select exactly 2 batters.")
                return

            # Assign to match_data
            self.match_data.striker = selected_batters[0]
            self.match_data.non_striker = selected_batters[1]

            # Initialize batter stats
            for pid in selected_batters:
                self.match_data.batters[pid] = {
                    'runs': 0,
                    'balls': 0,
                    'fours': 0,
                    'sixes': 0,
                    'status': 'not out'
                }

            # Start partnership
            self._start_new_partnership(opening=True)

            # Update display
            self.update_display()

            window.destroy()

        ttkb.Button(window, text="Confirm", command=confirm_selection, bootstyle="success").pack(pady=10)

    def clear_stats_trees(self):
        for tree in [self.batters_tree, self.bowlers_tree]:
            for item in tree.get_children():
                tree.delete(item)

    def set_submit_enabled(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.submit_btn.configure(state=state)

if __name__ == "__main__":
    root = ttkb.Window(themename='cyborg')
    app = CricketAnalysisApp(root)
    root.mainloop()
