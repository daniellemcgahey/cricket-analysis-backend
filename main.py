from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import io
import base64
from reportlab.lib import colors
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, TableStyle, PageBreak, Image, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib.enums import TA_CENTER
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal, Tuple
from collections import defaultdict, Counter
import os
import sqlite3
import math
import statistics
from datetime import datetime, timedelta


import sys
print(sys.path)

DB_PATH = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
app = FastAPI()

# If not already there:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="/tmp"), name="static")

class ColorSquare(Flowable):
    def __init__(self, fill_color, size=8):
        super().__init__()
        self.fill_color = fill_color
        self.size = size

    def draw(self):
        self.canv.setStrokeColor(colors.black)
        self.canv.setFillColor(self.fill_color)
        self.canv.rect(0, 0, self.size, self.size, fill=1, stroke=1)

class ComparisonPayload(BaseModel):
    country1: str
    country2: str
    tournaments: List[str]
    selected_stats: List[str]
    selected_phases: List[str]
    bowler_type: List[str]
    bowling_arm: List[str]
    teamCategory: str
    selectedMatches: List[int]

class CompareOverTournamentPayload(BaseModel):
    country: str
    tournaments: List[str]
    selected_stats: List[str]
    selected_phases: List[str]
    bowler_type: List[str]
    bowling_arm: List[str]
    teamCategory: str
    selectedMatches: List[int]

class ComparePlayerOverTournamentPayload(BaseModel):
    player_id: int
    tournaments: List[str]
    selected_stats: List[str]
    selected_phases: List[str]
    bowler_type: List[str]
    bowling_arm: List[str]
    teamCategory: str
    selectedMatches: List[int]

class PressurePayload(BaseModel):
    country1: str
    country2: str
    tournaments: List[str]
    selectedPhases: List[str]
    selectedMatches: List[int]
    allMatchesSelected: bool
    teamCategory: str

class WagonWheelPayload(BaseModel):
    country1: str
    country2: str
    tournaments: List[str]
    selectedPhases: List[str]
    selectedMatches: List[int]
    allMatchesSelected: bool
    perspective: str
    selectedBowlingArms: Optional[List[str]] = []
    selectedBowlerTypes: Optional[List[str]] = []
    selectedBattingHands: Optional[List[str]] = [] 
    teamCategory: str
    selectedLengths: Optional[List[str]] = []

class PitchMapPayload(BaseModel):
    country1: str
    country2: str
    tournaments: List[str]
    selectedPhases: List[str]
    selectedMatches: List[int]
    allMatchesSelected: bool
    selectedBowlingArms: Optional[List[str]] = []
    selectedBowlerTypes: Optional[List[str]] = []
    selectedBattingHands: Optional[List[str]] = [] 
    teamCategory: str

class TacticalMatchupPayload(BaseModel):
    batting_team: str
    bowling_team: str
    selected_phases: Optional[List[str]] = []
    team_category: str
    analyze_role: Optional[str] = "batting"

class SimulateMatchPayload(BaseModel):
    team_a_name: str
    team_b_name: str
    team_a_players: List[int]  # player_ids in batting order
    team_b_players: List[int]
    max_overs: int = 20
    team_category: str
    simulations: int = 1  # can allow multi-run sims later

class PlayerBattingAnalysisPayload(BaseModel):
    player_ids: List[int]
    tournaments: List[str]
    team_category: str
    bowling_arm: Optional[List[str]] = None
    bowling_style: Optional[List[str]] = None
    lengths: Optional[List[str]] = None

class PlayerBowlingAnalysisPayload(BaseModel):
    player_ids: List[int]
    team_category: str
    tournaments: List[str]
    bowling_arm: List[str]
    bowling_style: List[str]
    lengths: List[str]

class TrendAnalysisPayload(BaseModel):
    player_id: int
    tournaments: Optional[List[str]] = []
    team_category: str

class TrendAnalysisBowlingPayload(BaseModel):
    player_id: int
    tournaments: Optional[List[str]] = []
    team_category: str

class MatchScorecardPayload(BaseModel):
    team_category: str
    tournament: str
    match_id: int  

class MatchPressurePayload(BaseModel):
    team_category: str
    tournament: str
    match_id: int

class MatchPartnershipsPayload(BaseModel):
    team_category: str
    tournament: str
    match_id: int

class PlayerDetailedBattingPayload(BaseModel):
    team_category: str
    tournaments: List[str]
    player_ids: List[int]
    match_id: Optional[int] = None
    bowling_arm: List[str] = []  
    bowling_style: List[str] = []  
    lengths: Optional[List[str]] = None

class PlayerIntentSummaryPayload(BaseModel):
    player_ids: List[int]
    tournaments: List[str]
    team_category: str
    match_id: Optional[int] = None
    bowling_arm: List[str] = []
    bowling_style: List[str] = []
    lengths: Optional[List[str]] = None

class PlayerDetailedBowlingPayload(BaseModel):
    team_category: str
    tournaments: List[str]
    player_ids: List[int]
    match_id: Optional[int] = None
    batting_hand: List[str] = []
    bowling_style: List[str] = []
    lengths: Optional[List[str]] = None

class MatchBallByBallPayload(BaseModel):
    team_category: Optional[str] = None
    tournament: Optional[str] = None
    match_id: int

class GamePlanPayload(BaseModel):
    player_ids: list[int]  # opposition batter IDs
    bowler_ids: list[int]  # our selected bowler IDs
    team_category: str
    opponent_country: str

class MatchupDetailPayload(BaseModel):
    player_id: int
    team_category: str

class TournamentBowlingLeadersPayload(BaseModel):
    team_category: str
    tournament: str
    countries: List[str]

class TournamentFieldingLeadersPayload(BaseModel):
    team_category: str
    tournament: str
    countries: List[str]

class CoachPackRequest(BaseModel):
    match_id: int
    our_team_id: int          # country_id
    opponent_team_id: int     # country_id
    context: Literal["pre", "live", "post"] = "post"  # drive which sections to include
    top_n_matchups: int = 5
    min_balls_matchup: int = 12

class OppKeyPlayersPayload(BaseModel):
    team_category: str
    opponent_country: str
    min_balls: int = 40
    min_overs: float = 10.0

class OppositionStrengthsPayload(BaseModel):
    team_category: str
    opponent_country: str
    min_balls_style: int = 60
    min_balls_phase: int = 60
    min_balls_bowling: int = 120

class DoDontPayload(BaseModel):
    team_category: str                       # "Women", "Men", "U19 Women", ...
    opponent_country: str                    # e.g., "Rwanda Women"
    ground: Optional[str] = None             # e.g., "Kigali Oval"
    time_of_day: Optional[str] = None        # e.g., "Morning", "Afternoon" (depends on your data)
    min_balls_style: int = 120               # min balls vs a style for team-level inference
    min_balls_death_batter: int = 60         # min balls for an individual at death
    lookback_years: Optional[int] = 3        # future use; keeps simple for now

class DoDontResponse(BaseModel):
    do: Dict[str, Any]
    dont: Dict[str, Any]

def _db():
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.post("/compare")
def compare_countries(payload: ComparisonPayload):
    country1_stats = get_country_stats(
        country=payload.country1,
        tournaments=payload.tournaments,
        selected_stats=payload.selected_stats,
        selected_phases=payload.selected_phases,
        bowler_type=payload.bowler_type,
        bowling_arm=payload.bowling_arm,
        team_category=payload.teamCategory,
        selected_matches=payload.selectedMatches
    )

    country2_stats = get_country_stats(
    country=payload.country2,
    tournaments=payload.tournaments,
    selected_stats=payload.selected_stats,
    selected_phases=payload.selected_phases,
    bowler_type=payload.bowler_type,
    bowling_arm=payload.bowling_arm,
    team_category=payload.teamCategory,
    selected_matches=payload.selectedMatches
    )

    return {
        "country1": payload.country1,
        "country2": payload.country2,
        "selected_stats": payload.selected_stats,
        "selected_phases": payload.selected_phases,
        "bowler_type": payload.bowler_type,
        "bowling_arm": payload.bowling_arm,
        "country1_stats": country1_stats,
        "country2_stats": country2_stats
    }

@app.post("/compare_over_tournament")
def compare_over_tournament(payload: CompareOverTournamentPayload):
    result = {}

    for tournament in payload.tournaments:
        stats = get_country_stats(
            country=payload.country,
            tournaments=[tournament],  # Send one tournament at a time
            selected_stats=payload.selected_stats,
            selected_phases=payload.selected_phases,
            bowler_type=payload.bowler_type,
            bowling_arm=payload.bowling_arm,
            team_category=payload.teamCategory,
            selected_matches=payload.selectedMatches
        )
        result[tournament] = stats

    return {
        "country": payload.country,
        "tournaments": payload.tournaments,
        "stats_by_tournament": result
    }

@app.post("/compare_player_over_tournament")
def compare_player_over_tournament(payload: ComparePlayerOverTournamentPayload):
    result = {}

    for tournament in payload.tournaments:
        stats = get_player_stats(
            player_id=payload.player_id,
            tournaments=[tournament],  # one at a time
            selected_stats=payload.selected_stats,
            selected_phases=payload.selected_phases,
            bowler_type=payload.bowler_type,
            bowling_arm=payload.bowling_arm,
            team_category=payload.teamCategory,
            selected_matches=payload.selectedMatches
        )
        result[tournament] = stats

    return {
        "player_id": payload.player_id,
        "tournaments": payload.tournaments,
        "stats_by_tournament": result
    }

@app.post("/coach-pack")
def build_coach_pack(payload: CoachPackRequest):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ---- Match summary (you already do similar in generate_team_pdf_report) ----
    c.execute("""
      SELECT m.match_date, c1.country_name AS team_a, c2.country_name AS team_b, 
             m.result, m.adjusted_target, m.toss_winner, m.toss_decision
      FROM matches m
      JOIN countries c1 ON m.team_a=c1.country_id
      JOIN countries c2 ON m.team_b=c2.country_id
      WHERE m.match_id=?
    """, (payload.match_id,))
    ms = dict(c.fetchone() or {})

    # ---- KPIs & over medals (reuse your functions) ----
    kpis, medal_tallies_by_area = calculate_kpis(c, payload.match_id, payload.our_team_id, payload.our_team_id)
    over_medals = calculate_over_medals(c, payload.match_id, payload.our_team_id)

    # ---- Top favorable / avoid matchups using your /tactical-matchup-detailed logic style ----
    # Build batting-vs-bowler style pairs for both teams, then rank by a composite score.
    # Keep it simple: use your v of avg_rpb + dismissal_pct + dot% from /tactical-matchup-detailed.
    # NOTE: zero-change DB — this is just SQL + small Python math.
    def fetch_matchups(team_bat_id, team_bowl_id):
        c.execute("""
          SELECT be.batter_id, be.bowler_id,
                 SUM(CASE WHEN be.wides=0 THEN 1 ELSE 0 END) AS legal_balls,
                 SUM(be.runs + be.wides + be.no_balls + be.byes + be.leg_byes) AS runs,
                 SUM(CASE WHEN be.wides=0 AND be.no_balls=0 AND COALESCE(be.runs,0)=0
                           AND COALESCE(be.byes,0)=0 AND COALESCE(be.leg_byes,0)=0 
                      THEN 1 ELSE 0 END) AS dots,
                 SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS outs
          FROM ball_events be
          JOIN innings i ON be.innings_id=i.innings_id
          WHERE i.match_id=? AND i.batting_team=? AND i.bowling_team=?
          GROUP BY be.batter_id, be.bowler_id
        """, (payload.match_id, team_bat_id, team_bowl_id))
        rows = [dict(r) for r in c.fetchall()]
        out = []
        for r in rows:
            balls = r["legal_balls"] or 0
            if balls < payload.min_balls_matchup: 
                continue
            rpb = (r["runs"] / balls) if balls else 0.0
            dot_pct = (r["dots"]*100.0/balls) if balls else 0.0
            out_pct = (r["outs"]*100.0/balls) if balls else 0.0
            score = (out_pct/100.0) + (1.0/max(rpb, 0.1))  # same composite you use elsewhere
            out.append({**r, "rpb": round(rpb,2), "dot_pct": round(dot_pct,1), "dismissal_pct": round(out_pct,1), "score": round(score,3)})
        return sorted(out, key=lambda x: x["score"], reverse=True), sorted(out, key=lambda x: x["score"])

    # Favorable for us: bowlers who suppress their batters (high score = good for bowler)
    our_bowling_favorables, their_batting_favorables = fetch_matchups(team_bat_id=payload.opponent_team_id, team_bowl_id=payload.our_team_id)
    # Favorable for our batting: our batters vs their bowlers with LOW bowler score (invert)
    their_bowling_favorables, our_batting_favorables = fetch_matchups(team_bat_id=payload.our_team_id, team_bowl_id=payload.opponent_team_id)

    # ---- Intent bands from existing batting_intent_score ----
    c.execute("""
      SELECT be.batter_id,
             CASE 
               WHEN be.is_powerplay=1 THEN 'PP'
               WHEN be.is_death_overs=1 THEN 'DO'
               ELSE 'MO' END AS phase,
             CASE 
               WHEN be.batting_intent_score < 20 THEN '0-20'
               WHEN be.batting_intent_score < 40 THEN '20-40'
               WHEN be.batting_intent_score < 60 THEN '40-60'
               WHEN be.batting_intent_score < 80 THEN '60-80'
               ELSE '80-100' END AS band,
             COUNT(*) AS balls,
             SUM(be.runs + be.wides + be.no_balls + be.byes + be.leg_byes) AS runs,
             SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS outs
      FROM ball_events be
      JOIN innings i ON be.innings_id=i.innings_id
      WHERE i.match_id=? AND i.batting_team=?
      GROUP BY be.batter_id, phase, band
    """, (payload.match_id, payload.our_team_id))
    intent_rows = [dict(r) for r in c.fetchall()]
    intent_bands = []
    for r in intent_rows:
        balls = r["balls"] or 0
        sr = (r["runs"]*100.0/balls) if balls else 0.0
        dismiss_pct = (r["outs"]*100.0/balls) if balls else 0.0
        # heuristic: “green band” = top 2 SR bands with dismiss% not in top 2 highest
        intent_bands.append({**r, "sr": round(sr,1), "dismissal_pct": round(dismiss_pct,1)})

    # ---- 3 Do / 3 Don’t (auto text from your data) ----
    # Keep rules simple and deterministic so coaches trust them.
    do_list, dont_list = [], []

    # Do 1: Use our top bowling favorable matchup
    if our_bowling_favorables:
        f = our_bowling_favorables[0]
        do_list.append(f"Use our bowlers vs their batter #{f['batter_id']} (rpb {f['rpb']}, dismiss% {f['dismissal_pct']}%)")

    # Do 2: Intent band suggestion (pick the band with best SR where dismissal% <= team median)
    if intent_bands:
        # quick band per phase aggregate
        from statistics import median
        med = median([r["dismissal_pct"] for r in intent_bands])
        best = max([r for r in intent_bands if r["dismissal_pct"] <= med], key=lambda r: r["sr"], default=None)
        if best:
            do_list.append(f"Keep batter #{best['batter_id']} in {best['phase']} intent {best['band']} (SR {best['sr']}, dismiss {best['dismissal_pct']}%).")

    # Do 3: Bowlers’ PP/MO/DO phase with best dot% from your KPIs
    # (You can refine with your zone_effectiveness later.)
    c.execute("""
      SELECT be.bowler_id,
             CASE WHEN be.is_powerplay=1 THEN 'PP' WHEN be.is_death_overs=1 THEN 'DO' ELSE 'MO' END AS phase,
             ROUND(100.0*AVG(CASE WHEN be.wides=0 AND be.no_balls=0 
                     AND COALESCE(be.runs,0)=0 AND COALESCE(be.byes,0)=0 AND COALESCE(be.leg_byes,0)=0 THEN 1.0 ELSE 0.0 END),1) AS dot_pct
      FROM ball_events be JOIN innings i ON be.innings_id=i.innings_id
      WHERE i.match_id=? AND i.bowling_team=?
      GROUP BY be.bowler_id, phase
      ORDER BY dot_pct DESC LIMIT 1
    """, (payload.match_id, payload.our_team_id))
    top_dot = c.fetchone()
    if top_dot:
        do_list.append(f"Phase usage: #{top_dot['bowler_id']} in {top_dot['phase']} (dot {top_dot['dot_pct']}%).")

    # Don’t 1: Avoid our batting vs their best suppressor
    if their_bowling_favorables:
        avoid = their_bowling_favorables[0]
        dont_list.append(f"Avoid our batter #{avoid['batter_id']} vs their bowler #{avoid['bowler_id']} (rpb {avoid['rpb']}, dismiss% {avoid['dismissal_pct']}%).")
    # Don’t 2: PP boundaries conceded (from your KPI)
    # (We’ll scan your kpis for “PP Boundaries (Bowling)” > Bronze target)
    for k in kpis:
        if k["name"] == "PP Boundaries (Bowling)" and isinstance(k["targets"], dict):
            if k["actual"] > k["targets"]["Bronze"]:
                dont_list.append("Tighten PP boundary prevention—exceeded Bronze threshold.")
            break
    # Don’t 3: Extras if above target
    for k in kpis:
        if k["name"] == "Extras" and isinstance(k["targets"], dict):
            if k["actual"] > k["targets"]["Bronze"]:
                dont_list.append("Cut extras—above Bronze threshold.")
            break

    pack = {
        "match_summary": ms,
        "kpis": kpis,
        "medal_tallies_by_area": medal_tallies_by_area,
        "over_medals": over_medals,
        "favorable_bowling": our_bowling_favorables[:payload.top_n_matchups],
        "favorable_batting": our_batting_favorables[:payload.top_n_matchups],
        "intent_bands": intent_bands,
        "three_do": do_list[:3],
        "three_dont": dont_list[:3]
    }
    conn.close()
    return pack

@app.post("/wagon-wheel-comparison")
def wagon_wheel_comparison(payload: WagonWheelPayload):
    print("📨 Received Wagon Wheel Payload:", payload.dict())
    return get_wagon_wheel_data(payload)

@app.post("/pressure-analysis")
def pressure_analysis(payload: PressurePayload):
    print("✅ pressure_analysis route hit with payload:", payload.dict())  # Add this line
    return get_pressure_analysis(payload)

@app.get("/matches")
def get_matches(teamCategory: Optional[str] = None):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # include m.team_a and m.team_b as IDs alongside the country names
    base_query = """
        SELECT 
            m.match_id,
            t.tournament_name,
            m.team_a      AS team_a_id,
            c1.country_name AS team_a,
            m.team_b      AS team_b_id,
            c2.country_name AS team_b,
            m.match_date
        FROM matches m
        JOIN countries c1 ON m.team_a = c1.country_id
        JOIN countries c2 ON m.team_b = c2.country_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id
    """

    params = []
    if teamCategory:
        lc = teamCategory.lower()
        if lc == "training":
            base_query += " WHERE LOWER(c1.country_name) LIKE ? OR LOWER(c2.country_name) LIKE ?"
            params = ["%training%", "%training%"]
        else:
            base_query += """
                WHERE 
                  (c1.country_name LIKE ? AND LOWER(c1.country_name) NOT LIKE ?) OR
                  (c2.country_name LIKE ? AND LOWER(c2.country_name) NOT LIKE ?)
            """
            params = [f"%{teamCategory}", "%training%", f"%{teamCategory}", "%training%"]

    # add ORDER BY
    query = base_query + " ORDER BY m.match_date DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # map to JSON-friendly dicts, including the two new ID fields
    matches = []
    for row in rows:
        matches.append({
            "match_id":   row[0],
            "tournament": row[1],
            "team_a_id":  row[2],
            "team_a":     row[3],
            "team_b_id":  row[4],
            "team_b":     row[5],
            "match_date": row[6],
        })

    return matches

@app.get("/countries")
def get_countries(teamCategory: Optional[str] = None, tournament: Optional[str] = None):
    import sqlite3, os
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    query = """
        SELECT DISTINCT c.country_name
        FROM countries c
        JOIN matches m ON c.country_id = m.team_a OR c.country_id = m.team_b
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE 1 = 1
    """
    params = []

    if teamCategory:
        if teamCategory.lower() == "training":
            query += " AND LOWER(c.country_name) LIKE ?"
            params.append("%training%")
        else:
            query += " AND c.country_name LIKE ? AND LOWER(c.country_name) NOT LIKE ?"
            params.extend([f"%{teamCategory}", "%training%"])

    if tournament:
        query += " AND LOWER(t.tournament_name) = ?"
        params.append(tournament.lower())

    query += " ORDER BY c.country_name ASC"

    c.execute(query, params)
    countries = [row[0] for row in c.fetchall()]
    conn.close()
    return countries

@app.get("/tournaments")
def get_tournaments(teamCategory: Optional[str] = None):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if teamCategory:
        if teamCategory.lower() == "training":
            query = """
                SELECT DISTINCT t.tournament_name
                FROM tournaments t
                JOIN matches m ON m.tournament_id = t.tournament_id
                JOIN countries c1 ON m.team_a = c1.country_id
                JOIN countries c2 ON m.team_b = c2.country_id
                WHERE LOWER(c1.country_name) LIKE ? OR LOWER(c2.country_name) LIKE ?
                ORDER BY t.tournament_name ASC
            """
            c.execute(query, ("%training%", "%training%"))
        else:
            query = """
                SELECT DISTINCT t.tournament_name
                FROM tournaments t
                JOIN matches m ON m.tournament_id = t.tournament_id
                JOIN countries c1 ON m.team_a = c1.country_id
                JOIN countries c2 ON m.team_b = c2.country_id
                WHERE 
                    (c1.country_name LIKE ? AND LOWER(c1.country_name) NOT LIKE ?) OR 
                    (c2.country_name LIKE ? AND LOWER(c2.country_name) NOT LIKE ?)
                ORDER BY t.tournament_name ASC
            """
            c.execute(query, (f"%{teamCategory}", "%training%", f"%{teamCategory}", "%training%"))
    else:
        c.execute("SELECT tournament_name FROM tournaments ORDER BY tournament_name ASC")

    tournaments = [row["tournament_name"] for row in c.fetchall()]
    conn.close()
    return tournaments

@app.post("/pitch-map-comparison")
def pitch_map_comparison(payload: PitchMapPayload):
    return get_pitch_map_data(payload)

@app.post("/tactical-matchups")
def get_tactical_matchups(payload: TacticalMatchupPayload):
    print("📨 Tactical matchup request received:", payload.dict())

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ✅ Resolve team IDs
    cursor.execute("SELECT country_id FROM countries WHERE country_name = ?", (payload.batting_team,))
    my_team_id = cursor.fetchone()[0]

    cursor.execute("SELECT country_id FROM countries WHERE country_name = ?", (payload.bowling_team,))
    opp_team_id = cursor.fetchone()[0]

    # ✅ Phase filter (Powerplay, Middle Overs, Death Overs)
    phase_map = {
        "Powerplay": "be.is_powerplay = 1",
        "Middle Overs": "be.is_middle_overs = 1",
        "Death Overs": "be.is_death_overs = 1"
    }
    phase_clauses = [phase_map[p] for p in payload.selected_phases if p in phase_map]
    phase_filter = f"({' OR '.join(phase_clauses)})" if phase_clauses else "1=1"

    results = []

    # 🔄 FLIP LOGIC: Opposition Batting or Your Batting
    if payload.analyze_role == "opposition_batting":
        # 🔍 Analyze opposition batters vs YOUR bowling types
        cursor.execute("SELECT player_id, player_name FROM players WHERE country_id = ?", (opp_team_id,))
        batters = cursor.fetchall()

        # ✅ Your bowling profiles
        cursor.execute("""
            SELECT DISTINCT bowling_style, bowling_arm
            FROM players
            WHERE country_id = ? AND bowling_style IS NOT NULL
        """, (my_team_id,))
        bowler_profiles = cursor.fetchall()

        for batter in batters:
            batter_id = batter["player_id"]
            batter_name = batter["player_name"]

            for profile in bowler_profiles:
                bowler_type = profile["bowling_style"]
                bowler_arm = profile["bowling_arm"]

                query = f"""
                    SELECT
                        COUNT(*) AS balls_faced,
                        SUM(CASE WHEN be.runs = 0 AND be.extras = 0 THEN 1 ELSE 0 END) AS dot_balls,
                        SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS dismissals,
                        SUM(be.runs) AS total_runs
                    FROM ball_events be
                    JOIN innings i ON be.innings_id = i.innings_id
                    JOIN matches m ON i.match_id = m.match_id
                    JOIN players bowl ON be.bowler_id = bowl.player_id
                    WHERE be.batter_id = ?
                      AND bowl.bowling_style = ?
                      AND bowl.bowling_arm = ?
                      AND {phase_filter}
                """
                cursor.execute(query, [batter_id, bowler_type, bowler_arm])
                row = cursor.fetchone()
                balls = row["balls_faced"] or 0
                dots = row["dot_balls"] or 0
                outs = row["dismissals"] or 0
                runs = row["total_runs"] or 0

                if balls < 5:
                    continue

                results.append({
                    "batter": batter_name,
                    "bowler_type": bowler_type,
                    "bowling_arm": bowler_arm,
                    "balls_faced": balls,
                    "dot_rate": round(dots * 100 / balls, 1),
                    "dismissal_rate": round(outs * 100 / balls, 1),
                    "avg_runs_per_ball": round(runs / balls, 2),
                    "grade": "Unfavorable" if runs / balls < 0.8 else "Favorable" if runs / balls > 1.2 else "Neutral"
                })

    else:
        # 🔍 Analyze your batters vs opposition bowling types
        cursor.execute("SELECT player_id, player_name FROM players WHERE country_id = ?", (my_team_id,))
        batters = cursor.fetchall()

        # ✅ Opposition bowling profiles
        cursor.execute("""
            SELECT DISTINCT bowling_style, bowling_arm
            FROM players
            WHERE country_id = ? AND bowling_style IS NOT NULL
        """, (opp_team_id,))
        bowler_profiles = cursor.fetchall()

        for batter in batters:
            batter_id = batter["player_id"]
            batter_name = batter["player_name"]

            for profile in bowler_profiles:
                bowler_type = profile["bowling_style"]
                bowler_arm = profile["bowling_arm"]

                query = f"""
                    SELECT
                        COUNT(*) AS balls_faced,
                        SUM(CASE WHEN be.runs = 0 AND be.extras = 0 THEN 1 ELSE 0 END) AS dot_balls,
                        SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS dismissals,
                        SUM(be.runs) AS total_runs
                    FROM ball_events be
                    JOIN innings i ON be.innings_id = i.innings_id
                    JOIN matches m ON i.match_id = m.match_id
                    JOIN players bowl ON be.bowler_id = bowl.player_id
                    WHERE be.batter_id = ?
                      AND bowl.bowling_style = ?
                      AND bowl.bowling_arm = ?
                      AND {phase_filter}
                """
                cursor.execute(query, [batter_id, bowler_type, bowler_arm])
                row = cursor.fetchone()
                balls = row["balls_faced"] or 0
                dots = row["dot_balls"] or 0
                outs = row["dismissals"] or 0
                runs = row["total_runs"] or 0

                if balls < 5:
                    continue

                results.append({
                    "batter": batter_name,
                    "bowler_type": bowler_type,
                    "bowling_arm": bowler_arm,
                    "balls_faced": balls,
                    "dot_rate": round(dots * 100 / balls, 1),
                    "dismissal_rate": round(outs * 100 / balls, 1),
                    "avg_runs_per_ball": round(runs / balls, 2),
                    "grade": "Favorable" if runs / balls > 1.2 and outs / balls < 0.1 else "Unfavorable" if runs / balls < 0.8 else "Neutral"
                })

    conn.close()

    if not results:
        return {"matchups": [], "message": "Not enough data to infer matchups for this combination."}

    return {"matchups": results}

@app.post("/simulate-match")
def simulate_match(payload: SimulateMatchPayload):
    import sqlite3
    import random
    from collections import defaultdict

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    def get_bowler_weights(bowler_ids):
        weights = {}
        for bowler_id in bowler_ids:
            cursor.execute("""
                SELECT COUNT(DISTINCT i.match_id) AS games,
                       COUNT(*) AS balls
                FROM ball_events be
                JOIN innings i ON be.innings_id = i.innings_id
                WHERE be.bowler_id = ?
            """, (bowler_id,))
            row = cursor.fetchone()
            if row["games"] > 0:
                avg_overs = row["balls"] / 6 / row["games"]
                weights[bowler_id] = avg_overs

        # Fallback: assign default weights if insufficient data
        if len(weights) < 5:
            for bowler_id in bowler_ids:
                if bowler_id not in weights:
                    weights[bowler_id] = 1.0

        return weights

    def get_player_name(player_id):
        cursor.execute("SELECT player_name FROM players WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()
        return row["player_name"] if row else "Unknown"

    def get_matchup_probs(batter_id, bowler_id, phase_column):
        cursor.execute(f"""
            SELECT
                COUNT(*) AS total_balls,
                SUM(CASE WHEN be.runs = 0 AND be.extras = 0 THEN 1 ELSE 0 END) AS dot_balls,
                SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS dismissals,
                SUM(be.runs) AS total_runs
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE be.batter_id = ?
              AND be.bowler_id = ?
              AND be.{phase_column} = 1
        """, (batter_id, bowler_id))

        row = cursor.fetchone()
        balls = row["total_balls"] or 0
        if balls < 5:
            return {"dot": 0.3, "dismissal": 0.1, "rpb": 1.0}

        return {
            "dot": row["dot_balls"] / balls,
            "dismissal": row["dismissals"] / balls,
            "rpb": row["total_runs"] / balls
        }

    def simulate_innings(batting_team, bowling_team, phase_map, max_overs):
        score = 0
        wickets = 0
        over_data = []

        batters = batting_team[:]
        dismissed = set()
        striker_idx = 0
        non_striker_idx = 1

        bowler_overs = defaultdict(int)
        bowler_weights = get_bowler_weights(bowling_team)
        available_bowlers = [b for b in bowling_team if b in bowler_weights]
        previous_bowler = None

        for over in range(max_overs):
            phase = (
                "Powerplay" if over < 6 else
                "Middle Overs" if over < 16 else
                "Death Overs"
            )

            eligible_bowlers = [b for b in available_bowlers if bowler_overs[b] < 4 and b != previous_bowler]
            if not eligible_bowlers:
                eligible_bowlers = [b for b in available_bowlers if bowler_overs[b] < 4]
            if not eligible_bowlers:
                break

            eligible_weights = [bowler_weights[b] for b in eligible_bowlers]
            bowler = random.choices(eligible_bowlers, weights=eligible_weights, k=1)[0]
            previous_bowler = bowler
            bowler_overs[bowler] += 1
            bowler_name = get_player_name(bowler)

            runs_this_over = 0
            wickets_this_over = 0

            for ball in range(6):
                if wickets == 10 or striker_idx >= len(batters):
                    break

                striker = batters[striker_idx]
                probs = get_matchup_probs(striker, bowler, phase_map[phase])
                outcome = random.random()

                if outcome < probs["dismissal"]:
                    wickets += 1
                    wickets_this_over += 1
                    dismissed.add(striker)
                    next_idx = max(striker_idx, non_striker_idx) + 1
                    while next_idx < len(batters) and batters[next_idx] in dismissed:
                        next_idx += 1
                    striker_idx = next_idx
                elif outcome < probs["dismissal"] + probs["dot"]:
                    pass
                else:
                    runs = round(probs["rpb"])
                    score += runs
                    runs_this_over += runs
                    if runs % 2 != 0:
                        striker_idx, non_striker_idx = non_striker_idx, striker_idx

            over_data.append({
                "over_number": over + 1,
                "bowler": bowler_name,
                "runs": runs_this_over,
                "wickets": wickets_this_over,
                "cumulative_score": score,
                "cumulative_wickets": wickets
            })

            striker_idx, non_striker_idx = non_striker_idx, striker_idx

        return score, wickets, over_data

    # Setup
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    phase_column_map = {
        "Powerplay": "is_powerplay",
        "Middle Overs": "is_middle_overs",
        "Death Overs": "is_death_overs"
    }

    sim_runs_a, sim_runs_b = [], []
    sim_overs_a, sim_overs_b = None, None
    sim_wkts_a, sim_wkts_b = 0, 0
    wins_a = wins_b = 0
    margin_runs_a = []
    margin_wkts_b = []

    for _ in range(payload.simulations):
        runs_a, wkts_a, overs_a = simulate_innings(payload.team_a_players, payload.team_b_players, phase_column_map, payload.max_overs)
        runs_b, wkts_b, overs_b = simulate_innings(payload.team_b_players, payload.team_a_players, phase_column_map, payload.max_overs)

        sim_runs_a.append(runs_a)
        sim_runs_b.append(runs_b)
        sim_overs_a, sim_overs_b = overs_a, overs_b
        sim_wkts_a += wkts_a
        sim_wkts_b += wkts_b

        if runs_a > runs_b:
            wins_a += 1
            margin_runs_a.append(runs_a - runs_b)
        elif runs_b > runs_a:
            wins_b += 1
            margin_wkts_b.append(10 - wkts_b)

    total = payload.simulations
    avg_a = round(sum(sim_runs_a) / total, 1)
    avg_b = round(sum(sim_runs_b) / total, 1)
    prob_a = round((wins_a / total) * 100, 1)
    prob_b = round((wins_b / total) * 100, 1)

    margin_a = f"{round(sum(margin_runs_a)/len(margin_runs_a), 1)} runs" if margin_runs_a else "N/A"
    margin_b = f"{round(sum(margin_wkts_b)/len(margin_wkts_b), 1)} wickets" if margin_wkts_b else "N/A"

    winner = (
        payload.team_a_name if avg_a > avg_b else
        payload.team_b_name if avg_b > avg_a else
        "Draw"
    )

    return {
        "team_a": {
            "name": payload.team_a_name,
            "average_score": avg_a,
            "win_probability": prob_a,
            "expected_margin": margin_a,
            "last_sim_overs": sim_overs_a,
            "wickets": sim_wkts_a
        },
        "team_b": {
            "name": payload.team_b_name,
            "average_score": avg_b,
            "win_probability": prob_b,
            "expected_margin": margin_b,
            "last_sim_overs": sim_overs_b,
            "wickets": sim_wkts_b
        },
        "winner": winner
    }

@app.post("/simulate-match-v2")
def simulate_match_v2(payload: SimulateMatchPayload):
    import sqlite3
    import random
    from collections import defaultdict
    import os

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ---- Team Strength Calculation ----
    def get_team_strengths(team_id):
        cursor.execute("""
            SELECT 
                AVG(be.runs) AS rpb,
                AVG(be.expected_runs) AS x_rpb,
                AVG(be.batting_bpi) AS batting_bpi,
                AVG(be.bowling_bpi) AS bowling_bpi
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.batting_team = ?
        """, (team_id,))
        bat_row = cursor.fetchone()

        cursor.execute("""
            SELECT 
                AVG(be.runs) AS rpb_conceded,
                AVG(be.expected_runs) AS x_rpb_conceded,
                AVG(be.bowling_bpi) AS bowling_bpi
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.bowling_team = ?
        """, (team_id,))
        bowl_row = cursor.fetchone()

        return {
            "batting_rpb": bat_row["rpb"] or 1.0,
            "expected_rpb": bat_row["x_rpb"] or 1.0,
            "batting_bpi": bat_row["batting_bpi"] or 0.0,
            "bowling_rpb": bowl_row["rpb_conceded"] or 1.0,
            "bowling_bpi": bowl_row["bowling_bpi"] or 0.0,
        }

    cursor.execute("SELECT AVG(runs) FROM ball_events WHERE runs IS NOT NULL")
    global_rpb = cursor.fetchone()[0] or 1.0

    team_a_id = payload.team_a_players[0]  # any player ID from the team
    team_b_id = payload.team_b_players[0]

    team_a = get_team_strengths(team_a_id)
    team_b = get_team_strengths(team_b_id)

    # ---- Supporting Utilities ----
    def get_phase(over, max_overs):
        if over < max_overs * 0.3:
            return "is_powerplay"
        elif over < max_overs * 0.8:
            return "is_middle_overs"
        else:
            return "is_death_overs"

    def get_bowler_weights(bowler_ids):
        weights = {}
        for bowler_id in bowler_ids:
            cursor.execute("""
                SELECT COUNT(DISTINCT i.match_id) AS games,
                       COUNT(*) AS balls
                FROM ball_events be
                JOIN innings i ON be.innings_id = i.innings_id
                WHERE be.bowler_id = ?
            """, (bowler_id,))
            row = cursor.fetchone()
            if row["games"] > 0:
                weights[bowler_id] = row["balls"] / 6 / row["games"]
        for bowler_id in bowler_ids:
            if bowler_id not in weights:
                weights[bowler_id] = 1.0
        return weights

    def get_player_name(player_id):
        cursor.execute("SELECT player_name FROM players WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()
        return row["player_name"] if row else f"Player {player_id}"

    def get_outcome_distribution(batter_id, bowler_id, phase_column):
        cursor.execute(f"""
            SELECT
                be.runs AS runs,
                be.wides, be.no_balls,
                be.dismissal_type
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE be.batter_id = ?
              AND be.bowler_id = ?
              AND be.{phase_column} = 1
        """, (batter_id, bowler_id))
        events = cursor.fetchall()

        outcomes = []
        for row in events:
            if row["dismissal_type"]:
                outcomes.append(("WICKET", 0))
            elif row["wides"]:
                outcomes.append(("WIDE", row["wides"]))
            elif row["no_balls"]:
                outcomes.append(("NO_BALL", row["no_balls"]))
            else:
                outcomes.append(("RUN", row["runs"]))

        if len(outcomes) < 5:
            outcomes = [("RUN", 0)] * 3 + [("RUN", 1)] * 4 + [("RUN", 2)] * 2 + [("WICKET", 0)]

        return outcomes

    # ---- Simulation Logic ----
    def simulate_innings(batting_team, bowling_team, batting_strength, bowling_strength, max_overs):
        score = 0
        wickets = 0
        over_data = []

        batters = batting_team[:]
        striker_idx = 0
        non_striker_idx = 1
        dismissed = set()

        bowler_overs = defaultdict(int)
        bowler_weights = get_bowler_weights(bowling_team)
        available_bowlers = list(bowler_weights.keys())
        prev_bowler = None

        batting_multiplier = batting_strength["batting_rpb"] / global_rpb
        bowling_multiplier = global_rpb / (bowling_strength["bowling_rpb"] or 1.0)

        for over in range(max_overs):
            if wickets >= 10 or striker_idx >= len(batters):
                break

            phase_col = get_phase(over, max_overs)
            eligible_bowlers = [b for b in available_bowlers if bowler_overs[b] < 4 and b != prev_bowler]
            if not eligible_bowlers:
                eligible_bowlers = [b for b in available_bowlers if bowler_overs[b] < 4]
            if not eligible_bowlers:
                break

            weights = [bowler_weights[b] for b in eligible_bowlers]
            bowler = random.choices(eligible_bowlers, weights=weights, k=1)[0]
            bowler_name = get_player_name(bowler)
            prev_bowler = bowler
            bowler_overs[bowler] += 1

            runs_this_over = 0
            wickets_this_over = 0
            legal_deliveries = 0

            while legal_deliveries < 6:
                if wickets >= 10 or striker_idx >= len(batters):
                    break

                striker = batters[striker_idx]
                dist = get_outcome_distribution(striker, bowler, phase_col)
                outcome, value = random.choice(dist)

                # Apply matchup scaling
                if outcome == "RUN":
                    scaled = round(value * batting_multiplier * bowling_multiplier)
                    score += scaled
                    runs_this_over += scaled
                    if scaled % 2 == 1:
                        striker_idx, non_striker_idx = non_striker_idx, striker_idx
                    legal_deliveries += 1
                elif outcome == "WICKET":
                    wicket_prob = 1.0 / (batting_multiplier * bowling_multiplier)
                    if random.random() < min(wicket_prob, 0.6):
                        wickets += 1
                        wickets_this_over += 1
                        dismissed.add(striker)
                        next_idx = max(striker_idx, non_striker_idx) + 1
                        while next_idx < len(batters) and batters[next_idx] in dismissed:
                            next_idx += 1
                        striker_idx = next_idx
                    legal_deliveries += 1
                elif outcome == "WIDE" or outcome == "NO_BALL":
                    score += value
                    runs_this_over += value
                    # no delivery counted

            over_data.append({
                "over": over + 1,
                "bowler": bowler_name,
                "runs": runs_this_over,
                "wickets": wickets_this_over,
                "total_score": score,
                "total_wickets": wickets
            })

            striker_idx, non_striker_idx = non_striker_idx, striker_idx

        return score, wickets, over_data

    # ---- Run Simulations ----
    sim_runs_a, sim_runs_b = [], []
    sim_overs_a, sim_overs_b = None, None
    sim_wkts_a, sim_wkts_b = 0, 0
    wins_a = wins_b = 0
    margin_runs_a = []
    margin_wkts_b = []

    for _ in range(payload.simulations):
        runs_a, wkts_a, overs_a = simulate_innings(
            payload.team_a_players, payload.team_b_players, team_a, team_b, payload.max_overs)
        runs_b, wkts_b, overs_b = simulate_innings(
            payload.team_b_players, payload.team_a_players, team_b, team_a, payload.max_overs)

        sim_runs_a.append(runs_a)
        sim_runs_b.append(runs_b)
        sim_overs_a = overs_a
        sim_overs_b = overs_b
        sim_wkts_a += wkts_a
        sim_wkts_b += wkts_b

        if runs_a > runs_b:
            wins_a += 1
            margin_runs_a.append(runs_a - runs_b)
        elif runs_b > runs_a:
            wins_b += 1
            margin_wkts_b.append(10 - wkts_b)

    total = payload.simulations
    avg_a = round(sum(sim_runs_a) / total, 1)
    avg_b = round(sum(sim_runs_b) / total, 1)
    prob_a = round((wins_a / total) * 100, 1)
    prob_b = round((wins_b / total) * 100, 1)

    margin_a = f"{round(sum(margin_runs_a)/len(margin_runs_a), 1)} runs" if margin_runs_a else "N/A"
    margin_b = f"{round(sum(margin_wkts_b)/len(margin_wkts_b), 1)} wickets" if margin_wkts_b else "N/A"

    winner = (
        payload.team_a_name if avg_a > avg_b else
        payload.team_b_name if avg_b > avg_a else
        "Draw"
    )

    return {
        "team_a": {
            "name": payload.team_a_name,
            "average_score": avg_a,
            "win_probability": prob_a,
            "expected_margin": margin_a,
            "last_sim_overs": sim_overs_a,
            "wickets": sim_wkts_a
        },
        "team_b": {
            "name": payload.team_b_name,
            "average_score": avg_b,
            "win_probability": prob_b,
            "expected_margin": margin_b,
            "last_sim_overs": sim_overs_b,
            "wickets": sim_wkts_b
        },
        "winner": winner
    }

@app.get("/team-players")
def get_players_for_team(country_name: str, team_category: Optional[str] = None):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if team_category:
        if team_category.lower() == "training":
            cursor.execute("""
                SELECT p.player_id, p.player_name, p.bowling_arm, p.bowling_style
                FROM players p
                JOIN countries c ON p.country_id = c.country_id
                WHERE c.country_name = ? AND LOWER(c.country_name) LIKE ?
                ORDER BY p.player_name
            """, (country_name, "%training%"))
        else:
            cursor.execute("""
                SELECT p.player_id, p.player_name, p.bowling_arm, p.bowling_style
                FROM players p
                JOIN countries c ON p.country_id = c.country_id
                WHERE c.country_name = ? AND LOWER(c.country_name) NOT LIKE ?
                ORDER BY p.player_name
            """, (country_name, "%training%"))
    else:
        cursor.execute("""
            SELECT p.player_id, p.player_name, p.bowling_arm, p.bowling_style
            FROM players p
            JOIN countries c ON p.country_id = c.country_id
            WHERE c.country_name = ?
            ORDER BY p.player_name
        """, (country_name,))
    
    players = [{
        "id": row[0],
        "name": row[1],
        "bowling_arm": row[2],
        "bowling_style": row[3]
    } for row in cursor.fetchall()]
    conn.close()
    return players

@app.get("/players")
def get_players_by_team_category(team_category: str):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if team_category.lower() == "training":
        cursor.execute("""
            SELECT player_id, player_name
            FROM players p
            JOIN countries c ON p.country_id = c.country_id
            WHERE LOWER(c.country_name) LIKE ?
            ORDER BY player_name
        """, ("%training%",))
    else:
        cursor.execute("""
            SELECT player_id, player_name
            FROM players p
            JOIN countries c ON p.country_id = c.country_id
            WHERE c.country_name LIKE ? AND LOWER(c.country_name) NOT LIKE ?
            ORDER BY player_name
        """, (f"%{team_category}", "%training%"))

    players = [{"player_id": row[0], "player_name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return players

@app.post("/player-batting-analysis")
def player_batting_analysis(payload: PlayerBattingAnalysisPayload):
    import os
    import sqlite3

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 🎯 Resolve tournament IDs
    tournament_ids = []
    tournament_filter = ""
    if payload.tournaments:
        placeholders = ",".join(["?"] * len(payload.tournaments))
        cursor.execute(f"""
            SELECT tournament_id FROM tournaments
            WHERE tournament_name IN ({placeholders})
        """, payload.tournaments)
        tournament_ids = [row["tournament_id"] for row in cursor.fetchall()]
        if tournament_ids:
            tournament_filter = f" AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    # 🌎 Resolve country name using the first player ID
    cursor.execute("SELECT country_id FROM players WHERE player_id = ?", (payload.player_ids[0],))
    row = cursor.fetchone()
    if not row:
        return {"overall": [], "partnerships": [], "ten_ball": [], "by_position": [], "wagon_wheel": []}
    country_id = row["country_id"]

    cursor.execute("SELECT country_name FROM countries WHERE country_id = ?", (country_id,))
    country_name = cursor.fetchone()["country_name"]

    # 🎯 Build filters
    def list_filter_sql(column, values):
        if values:
            placeholders = ",".join(["?"] * len(values))
            return f" AND {column} IN ({placeholders})", values
        return "", []

    bowling_arm_filter, bowling_arm_params = list_filter_sql("p.bowling_arm", payload.bowling_arm)
    bowling_style_filter, bowling_style_params = list_filter_sql("p.bowling_style", payload.bowling_style)

    length_filter = ""
    if payload.lengths:
        conditions = []
        for length in payload.lengths:
            if length == "Full Toss":
                conditions.append("be.pitch_y BETWEEN 0.0 AND 0.1")
            elif length == "Yorker":
                conditions.append("be.pitch_y BETWEEN 0.1 AND 0.25")
            elif length == "Full":
                conditions.append("be.pitch_y BETWEEN 0.25 AND 0.4")
            elif length == "Good":
                conditions.append("be.pitch_y BETWEEN 0.4 AND 0.6")
            elif length == "Short":
                conditions.append("be.pitch_y BETWEEN 0.6 AND 1.0")
        if conditions:
            length_filter = f" AND ({' OR '.join(conditions)})"

    # 🎯 Build overall query
    player_placeholders = ",".join(["?"] * len(payload.player_ids))
    overall_params = (
        payload.player_ids +
        [country_name, country_name] +  # used for two subqueries
        tournament_ids +
        bowling_arm_params +
        bowling_style_params +
        payload.player_ids +
        [country_name] +
        tournament_ids +
        bowling_arm_params +
        bowling_style_params
    )


    cursor.execute(f"""
        WITH innings_summary AS (
            SELECT
                be.innings_id,
                i.batting_team,
                t.tournament_name,
                SUM(be.runs) AS runs,
                MAX(
                CASE
                    WHEN be.dismissed_player_id = be.batter_id
                    AND be.dismissal_type IS NOT NULL
                    AND LOWER(be.dismissal_type) != 'not out'
                    THEN 1 ELSE 0
                END
                ) AS dismissed

            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN tournaments t ON m.tournament_id = t.tournament_id
            JOIN players p ON be.bowler_id = p.player_id
            WHERE be.batter_id IN ({player_placeholders})
              {tournament_filter}
              {bowling_arm_filter}
              {bowling_style_filter}
              {length_filter}
            GROUP BY be.innings_id
        ),
        high_scores AS (
            SELECT tournament_name, MAX(runs) AS high_score
            FROM innings_summary
            WHERE batting_team = ?
            GROUP BY tournament_name
        ),
        hs_dismissals AS (
            SELECT s.tournament_name, s.runs AS high_score, s.dismissed AS high_score_dismissed
            FROM innings_summary s
            JOIN high_scores hs ON s.tournament_name = hs.tournament_name AND s.runs = hs.high_score
            WHERE s.batting_team = ?
        )
        SELECT 
            t.tournament_name,
            COUNT(DISTINCT i.innings_id) AS innings,
            SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END) AS balls_faced,
            SUM(be.runs) AS total_runs,
            SUM(
                CASE
                    WHEN be.wides = 0 AND be.runs = 0 THEN 1
                    ELSE 0
                END
                ) AS dots,
            SUM(CASE WHEN be.runs = 1 THEN 1 ELSE 0 END) AS ones,
            SUM(CASE WHEN be.runs = 2 THEN 1 ELSE 0 END) AS twos,
            SUM(CASE WHEN be.runs = 3 THEN 1 ELSE 0 END) AS threes,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            SUM(
                CASE
                    WHEN be.dismissed_player_id = be.batter_id
                    AND be.dismissal_type IS NOT NULL
                    AND LOWER(be.dismissal_type) != 'not out'
                    THEN 1 ELSE 0
                END
                ) AS dismissals,
            ROUND(SUM(be.runs) * 1.0 / COUNT(*), 2) AS rpb,
            ROUND(AVG(be.batting_intent_score), 2) AS avg_intent,
            hs.high_score,
            hs.high_score_dismissed

        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        JOIN players p ON be.bowler_id = p.player_id
        LEFT JOIN hs_dismissals hs ON hs.tournament_name = t.tournament_name
        WHERE be.batter_id IN ({player_placeholders})
          AND i.batting_team = ?
          {tournament_filter}
          {bowling_arm_filter}
          {bowling_style_filter}
          {length_filter}
        GROUP BY t.tournament_name
    """, overall_params)
    overall_stats = cursor.fetchall()

    # 🎯 Partnership stats
    cursor.execute(f"""
        SELECT p.start_wicket, p.runs, p.balls,
               CASE WHEN p.unbeaten = 1 THEN 1 ELSE 0 END AS unbeaten,
               c.country_name AS opponent,
               t.tournament_name,
               m.match_date
        FROM partnerships p
        JOIN innings i ON p.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        LEFT JOIN countries c ON p.opponent_team = c.country_name
        LEFT JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE (p.batter1_id IN ({player_placeholders}) OR p.batter2_id IN ({player_placeholders}))
          AND i.batting_team = ?
          {tournament_filter}
        ORDER BY p.runs DESC, p.balls ASC
        LIMIT 5
    """, payload.player_ids + payload.player_ids + [country_name] + tournament_ids)
    partnership_stats = cursor.fetchall()

    # 🎯 10-ball segments
    cursor.execute(f"""
        SELECT be.innings_id, be.ball_id, be.runs, be.dismissal_type, be.batting_intent_score
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.batter_id IN ({player_placeholders})
          AND i.batting_team = ?
          {tournament_filter}
        ORDER BY be.innings_id, be.ball_id
    """, payload.player_ids + [country_name] + tournament_ids)
    rows = cursor.fetchall()

    segment_stats = {}
    current_innings = None
    ball_count = 0
    for row in rows:
        innings_id = row["innings_id"]
        runs = row["runs"]
        dismissal = (row["dismissal_type"] or "").lower()
        intent = row["batting_intent_score"] or 0
        if current_innings != innings_id:
            current_innings = innings_id
            ball_count = 0
        segment = (ball_count // 10) * 10
        label = f"{segment}-{segment+9}"
        if label not in segment_stats:
            segment_stats[label] = {"runs": 0, "balls": 0, "scoring": 0, "dismissals": 0, "intent_total": 0}
        seg = segment_stats[label]
        seg["runs"] += runs
        seg["balls"] += 1
        seg["intent_total"] += intent
        if runs > 0:
            seg["scoring"] += 1
        if dismissal and dismissal != "not out":
            seg["dismissals"] += 1
        ball_count += 1

    ten_ball_output = [
        {
            "Segment": label,
            "Balls Faced": seg["balls"],
            "Runs": seg["runs"],
            "Avg Runs per Ball": round(seg["runs"] / max(1, seg["balls"]), 2),
            "Scoring %": round((seg["scoring"] / seg["balls"]) * 100, 2),
            "Dismissal %": round((seg["dismissals"] / seg["balls"]) * 100, 2),
            "Avg Intent": round(seg["intent_total"] / max(1, seg["balls"]), 2)
        }
        for label, seg in sorted(segment_stats.items(), key=lambda kv: int(kv[0].split("-")[0]))
    ]

    # 🎯 Batting position breakdown
    bat_pos_params = (
        payload.player_ids +            # used in high_scores_raw
        [country_name] +                # used in high_scores_raw
        tournament_ids +                # used in high_scores_raw
        payload.player_ids +            # used in final SELECT
        [country_name] +                # used in final SELECT
        tournament_ids                  # used in final SELECT
    )

    cursor.execute(f"""
        WITH high_scores_raw AS (
            SELECT be.innings_id, be.batting_position, SUM(be.runs) AS runs,
                MAX(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS dismissed
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE be.batter_id IN ({','.join(['?'] * len(payload.player_ids))})
            AND i.batting_team = ?
            {f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})" if tournament_ids else ""}
            GROUP BY be.innings_id
        ),
        high_scores_pos AS (
            SELECT batting_position, MAX(runs) AS high_score
            FROM high_scores_raw GROUP BY batting_position
        ),
        hs_final AS (
            SELECT hs.batting_position, hs.runs AS high_score, hs.dismissed AS high_score_dismissed
            FROM high_scores_raw hs
            JOIN high_scores_pos hp ON hs.batting_position = hp.batting_position AND hs.runs = hp.high_score
        )
        SELECT 
            be.batting_position,
            COUNT(*) AS balls_faced,
            COUNT(DISTINCT i.innings_id) AS innings,
            SUM(be.runs) AS total_runs,
            SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS dismissals,
            ROUND(AVG(be.batting_intent_score), 2) AS avg_intent,
            SUM(CASE WHEN be.runs > 0 THEN 1 ELSE 0 END) AS scoring_balls,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            hf.high_score,
            hf.high_score_dismissed
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        LEFT JOIN hs_final hf ON be.batting_position = hf.batting_position
        WHERE be.batter_id IN ({','.join(['?'] * len(payload.player_ids))})
        AND i.batting_team = ?
        {f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})" if tournament_ids else ""}
        GROUP BY be.batting_position
        ORDER BY be.batting_position
    """, bat_pos_params)
    batting_position_stats = cursor.fetchall()

    # 🎯 Wagon wheel
    wagon_wheel_data = []
    for player_id in payload.player_ids:
        wagon_wheel_data += get_individual_wagon_wheel_data(player_id, country_name, tournament_ids)

    return {
        "overall": [dict(row) for row in overall_stats],
        "partnerships": [dict(row) for row in partnership_stats],
        "ten_ball": ten_ball_output,
        "by_position": [dict(row) for row in batting_position_stats],
        "wagon_wheel": wagon_wheel_data
    }

@app.post("/player-bowling-analysis")
def player_bowling_analysis(payload: PlayerBowlingAnalysisPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    tournament_ids = []
    tournament_filter = ""
    tournament_params = []

    if payload.tournaments:
        cursor.execute(f"""
            SELECT tournament_id FROM tournaments
            WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
        """, payload.tournaments)
        tournament_ids = [row["tournament_id"] for row in cursor.fetchall()]
        if tournament_ids:
            tournament_filter = f" AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"
            tournament_params = tournament_ids

    cursor.execute("SELECT country_id FROM players WHERE player_id = ?", (payload.player_ids[0],))
    country_row = cursor.fetchone()
    if not country_row:
        return {"error": "Country not found for player."}
    
    cursor.execute("SELECT country_name FROM countries WHERE country_id = ?", (country_row["country_id"],))
    selected_country_name = cursor.fetchone()["country_name"]

    # === Overall Bowling Stats ===
    print("📌 Starting Overall Bowling Stats Calculation")
    print("👤 Bowler ID:", payload.player_ids)
    print("🌎 Bowling for Team:", selected_country_name)
    print("🏆 Tournament Filter Applied:", tournament_filter)
    print("🧮 Tournament Params:", tournament_params)

    # STEP 1: Raw aggregate stats per tournament (no best bowling join here)
    cursor.execute(f"""
        SELECT 
            t.tournament_name,
            COUNT(DISTINCT i.innings_id) AS innings,
            SUM(CASE WHEN json_extract(be.extras, '$.wides') = 0 AND json_extract(be.extras, '$.no_balls') = 0 THEN 1 ELSE 0 END) AS balls,
            SUM(be.runs) AS runs,
            SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS wickets,
            SUM(be.dot_balls) AS dots,
            SUM(be.wides) AS wides,
            SUM(be.no_balls) AS no_balls,
            SUM(be.expected_runs) AS expected_runs,
            SUM(be.expected_wicket) AS expected_wicket
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})
        AND i.bowling_team = ?
        {tournament_filter}
        GROUP BY t.tournament_name
    """, payload.player_ids + [selected_country_name] + tournament_params)

    raw_stats = cursor.fetchall()
    print("📊 Raw Overall Bowling Results:")
    for row in raw_stats:
        print(dict(row))

    # STEP 2: Get best bowling figures per tournament (wickets DESC, then runs ASC)
    cursor.execute(f"""
        WITH per_innings AS (
            SELECT
                t.tournament_name,
                be.innings_id,
                SUM(be.runs) AS runs,
                SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS wickets
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN tournaments t ON m.tournament_id = t.tournament_id
            WHERE be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})
            AND i.bowling_team = ?
            {tournament_filter}
            GROUP BY be.innings_id
        ),
        ranked_best AS (
            SELECT *,
                RANK() OVER (
                    PARTITION BY tournament_name
                    ORDER BY wickets DESC, runs ASC
                ) AS rnk
            FROM per_innings
        ),
        best_bowling AS (
            SELECT tournament_name, wickets, runs
            FROM ranked_best
            WHERE rnk = 1
        )

        SELECT * FROM best_bowling
    """, payload.player_ids + [selected_country_name] + tournament_params)

    best_figures_raw = cursor.fetchall()
    print("🎯 Best Bowling Figures Raw:")
    for row in best_figures_raw:
        print(dict(row))

    best_figures = {
        row["tournament_name"]: f"{row['wickets']}-{row['runs']}" for row in best_figures_raw
    }

    # STEP 3: Final formatting
    overall = []
    for row in raw_stats:
        tname = row["tournament_name"]
        balls = row["balls"]
        overs = balls // 6 + (balls % 6) / 10
        econ = row["runs"] / (balls / 6) if balls else 0
        avg = row["runs"] / row["wickets"] if row["wickets"] else "–"
        sr = balls / row["wickets"] if row["wickets"] else "–"
        best = best_figures.get(tname, "–")
        overall.append({
            **dict(row),
            "overs": round(overs, 1),
            "econ": round(econ, 2),
            "avg": round(avg, 2) if isinstance(avg, float) else "–",
            "sr": round(sr, 2) if isinstance(sr, float) else "–",
            "best": best,
            "expected_runs": row["expected_runs"] or 0,
            "expected_wicket": row["expected_wicket"] or 0
        })




    # === Best Performances (Top Wicket Hauls) ===
    print("📌 Fetching best bowling performances...")

    cursor.execute(f"""
        WITH ranked_innings AS (
            SELECT 
                t.tournament_name,
                be.innings_id,
                SUM(be.runs) AS runs_conceded,
                SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS wickets,
                SUM(CASE WHEN json_extract(be.extras, '$.wides') = 0 AND json_extract(be.extras, '$.no_balls') = 0 THEN 1 ELSE 0 END) AS balls_bowled,
                SUM(be.dot_balls) AS dots,
                SUM(be.wides) AS wides,
                SUM(be.no_balls) AS no_balls,
                m.match_date,
                i.batting_team AS opponent,
                RANK() OVER (
                    PARTITION BY t.tournament_name
                    ORDER BY 
                        SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) DESC,
                        SUM(be.runs) ASC
                ) AS rank
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN tournaments t ON m.tournament_id = t.tournament_id
            WHERE be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})
            AND i.bowling_team = ?
            {tournament_filter}
            GROUP BY t.tournament_name, be.innings_id
        )
        SELECT *
        FROM ranked_innings
        WHERE rank = 1
        ORDER BY wickets DESC, runs_conceded ASC
        LIMIT 5
    """, payload.player_ids + [selected_country_name] + tournament_params)


    best_performances = []
    for row in cursor.fetchall():
        balls = row["balls_bowled"]
        overs = balls // 6 + (balls % 6) / 10
        best_performances.append({
            **dict(row),
            "overs": round(overs, 1)
        })


    print("🎯 Top 5 Bowling Performances:")
    for perf in best_performances:
        print(perf)



    # === Phase Stats ===
    cursor.execute(f"""
        SELECT 
            SUM(CASE WHEN be.is_powerplay = 1 THEN 1 ELSE 0 END) AS powerplay_balls,
            SUM(CASE WHEN be.is_middle_overs = 1 THEN 1 ELSE 0 END) AS middle_balls,
            SUM(CASE WHEN be.is_death_overs = 1 THEN 1 ELSE 0 END) AS death_balls,
            SUM(CASE WHEN be.is_powerplay = 1 THEN be.runs ELSE 0 END) AS powerplay_runs,
            SUM(CASE WHEN be.is_middle_overs = 1 THEN be.runs ELSE 0 END) AS middle_runs,
            SUM(CASE WHEN be.is_death_overs = 1 THEN be.runs ELSE 0 END) AS death_runs,
            SUM(CASE WHEN be.is_powerplay = 1 AND be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS powerplay_wkts,
            SUM(CASE WHEN be.is_middle_overs = 1 AND be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS middle_wkts,
            SUM(CASE WHEN be.is_death_overs = 1 AND be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS death_wkts
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})
        AND i.bowling_team = ? {tournament_filter}
    """, payload.player_ids + [selected_country_name] + tournament_params)

    row = cursor.fetchone()
    phase_stats = {
        "Powerplay": {"balls": row["powerplay_balls"], "runs": row["powerplay_runs"], "wickets": row["powerplay_wkts"]},
        "Middle": {"balls": row["middle_balls"], "runs": row["middle_runs"], "wickets": row["middle_wkts"]},
        "Death": {"balls": row["death_balls"], "runs": row["death_runs"], "wickets": row["death_wkts"]},
    }
    print("📊 Phase Stats Raw:", dict(row) if row else "No data returned from query")

    # === Spell-Over Stats ===
    cursor.execute(f"""
        SELECT innings_id, over_number, bowler_id
        FROM ball_events be
        WHERE be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})
        GROUP BY innings_id, over_number, bowler_id
        ORDER BY innings_id, over_number
    """, (payload.player_ids))
    overs = cursor.fetchall()

    spells = []
    current_spell = []
    for row in overs:
        if not current_spell:
            current_spell.append(row)
        else:
            same_bowler = current_spell[-1]["bowler_id"] == row["bowler_id"]
            consecutive = int(row["over_number"]) == int(current_spell[-1]["over_number"]) + 1
            if same_bowler and consecutive:
                current_spell.append(row)
            else:
                spells.append(current_spell)
                current_spell = [row]
    if current_spell:
        spells.append(current_spell)

    tagged_overs = []
    for spell in spells:
        for i, over in enumerate(spell[:4]):
            tagged_overs.append({
                "innings_id": over["innings_id"],
                "bowler_id": over["bowler_id"],
                "over_number": over["over_number"],
                "spell_over_number": i + 1
            })

    spell_stats = defaultdict(lambda: {"balls": 0, "runs": 0, "wickets": 0, "dots": 0})
    for tag in tagged_overs:
        cursor.execute("""
            SELECT runs, dismissal_type, dot_balls
            FROM ball_events
            WHERE innings_id = ? AND bowler_id = ? AND CAST(over_number AS INTEGER) = ?
        """, (tag["innings_id"], tag["bowler_id"], int(tag["over_number"])))
        for ball in cursor.fetchall():
            s = spell_stats[tag["spell_over_number"]]
            s["balls"] += 1
            s["runs"] += ball["runs"] or 0
            s["dots"] += ball["dot_balls"] or 0
            if ball["dismissal_type"] not in (None, "", "not out"):
                s["wickets"] += 1

    def format_overs(balls):
        return f"{balls // 6}.{balls % 6}"

    by_spell_position = []
    for num in sorted(spell_stats.keys()):
        s = spell_stats[num]
        balls = s["balls"]
        overs = format_overs(balls)
        econ = s["runs"] / (balls / 6) if balls else 0
        avg = s["runs"] / s["wickets"] if s["wickets"] else "–"
        sr = balls / s["wickets"] if s["wickets"] else "–"
        
        by_spell_position.append({
            "spell_over": f"{num}{'st' if num == 1 else 'nd' if num == 2 else 'rd' if num == 3 else 'th'} Over",
            "overs": overs,
            "runs": s["runs"],
            "wickets": s["wickets"],
            "dots": s["dots"],
            "economy": round(econ, 2),
            "average": round(avg, 2) if s["wickets"] else "–",
            "strike_rate": round(sr, 2) if s["wickets"] else "–"
        })

    conn.close()
    return {
        "overall": overall,
        "best": best_performances,
        "phase": phase_stats,
        "by_spell_position": by_spell_position,

    }

@app.post("/player-trend-analysis")
def player_trend_analysis(payload: TrendAnalysisPayload):

    print("🎯 Incoming Trend Analysis Payload:")
    print(f"👤 Player ID: {payload.player_id}")
    print(f"🏆 Tournaments: {payload.tournaments}")
    print(f"📂 Team Category: {payload.team_category}")
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Resolve player team
    cursor.execute("SELECT country_id FROM players WHERE player_id = ?", (payload.player_id,))
    country_row = cursor.fetchone()
    if not country_row:
        return {"error": "Invalid player_id"}
    country_id = country_row["country_id"]

    cursor.execute("SELECT country_name FROM countries WHERE country_id = ?", (country_id,))
    country_name = cursor.fetchone()["country_name"]

    # Tournament filter
    tournament_ids = []
    tournament_filter = ""
    if payload.tournaments:
        cursor.execute(
            f"SELECT tournament_id FROM tournaments WHERE tournament_name IN ({','.join(['?']*len(payload.tournaments))})",
            payload.tournaments
        )
        tournament_ids = [r["tournament_id"] for r in cursor.fetchall()]
        if tournament_ids:
            tournament_filter = f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"
    print(f"✅ Resolved Tournament IDs: {tournament_ids}")

    # Final query args
    query_args = [payload.player_id, country_name] + tournament_ids

    # Batting History
    cursor.execute(f"""
        SELECT m.match_id, m.match_date, t.tournament_name, i.bowling_team AS opponent,
               SUM(be.runs) AS runs,
               COUNT(*) AS balls,
               ROUND(AVG(be.batting_intent_score), 2) AS avg_intent,
               SUM(CASE WHEN be.runs > 0 THEN 1 ELSE 0 END) AS scoring,
               SUM(CASE WHEN be.shot_type = 'Attacking' THEN 1 ELSE 0 END) AS attacking,
               SUM(CASE WHEN be.runs IN (4, 6) THEN be.runs ELSE 0 END) AS boundary_runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE be.batter_id = ?
        AND i.batting_team = ?
        {tournament_filter}
        GROUP BY m.match_id
        ORDER BY m.match_date
    """, query_args)

    history_rows = cursor.fetchall()
    history_data = []
    for idx, row in enumerate(history_rows):
        sr = (row["runs"] / row["balls"]) * 100 if row["balls"] else 0
        scoring_pct = (row["scoring"] / row["balls"]) * 100 if row["balls"] else 0
        boundary_pct = (row["boundary_runs"] / row["runs"]) * 100 if row["runs"] else 0
        history_data.append({
            "match_id": row["match_id"],
            "opponent": row["opponent"],
            "match_date": row["match_date"],
            "match_num": idx + 1,
            "runs": row["runs"],
            "balls": row["balls"],
            "intent": row["avg_intent"],
            "sr": round(sr, 2),
            "scoring_pct": round(scoring_pct, 2),
            "boundary_pct": round(boundary_pct, 2),
            "attacking_pct": round((row["attacking"] / row["balls"]) * 100, 2) if row["balls"] else 0
        })

    # Batting Intent per Over
    cursor.execute(f"""
        SELECT be.over_number, AVG(be.batting_intent_score) AS avg_intent
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.batter_id = ?
        AND i.batting_team = ?
        {tournament_filter}
        GROUP BY be.over_number
        ORDER BY be.over_number
    """, query_args)

    over_intent = [{"over": int(row["over_number"]), "intent": round(row["avg_intent"], 2)} for row in cursor.fetchall()]

    # Dismissal Trends (All-Time + Last 5 Matches)
    cursor.execute(f"""
        SELECT m.match_date, LOWER(be.dismissal_type) AS type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.dismissed_player_id = ?
        AND i.batting_team = ?
        AND be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out'
        {tournament_filter}
        ORDER BY m.match_date
    """, query_args)

    all_dismissals = cursor.fetchall()
    dismissal_counts = defaultdict(int)
    last_5 = defaultdict(int)

    recent = sorted(all_dismissals, key=lambda x: x["match_date"], reverse=True)[:5]
    for row in all_dismissals:
        dismissal_counts[row["type"]] += 1
    for row in recent:
        last_5[row["type"]] += 1
    print("📦 Final Trend Data Snapshot:")
    print(f"📉 History Points: {len(history_data)}")
    print(f"🔥 Overs with Intent Data: {[row['over'] for row in over_intent]}")
    if history_data:
        print("📈 Matchwise Trend Lines: Data available for SR, scoring%, boundary%, attacking%")
    else:
        print("📈 Matchwise Trend Lines: No match history available")
    print(f"☠️ Dismissals: {dismissal_counts}")


    match_trends = [
        {
            "match_num": row["match_num"],
            "opponent": row["opponent"],
            "match_date": row["match_date"],
            "scoring_shot_pct": row["scoring_pct"],
            "boundary_pct": row["boundary_pct"],
            "sr": row["sr"],
            "attacking_pct": row["attacking_pct"],
            "moving_avg_runs": None  # Will fill below
        }
        for row in history_data
    ]

    # Add 3-game moving average of runs
    for i in range(len(match_trends)):
        if i >= 2:
            last3 = history_data[i-2:i+1]
            avg = round(sum(r["runs"] for r in last3) / 3, 2)
            match_trends[i]["moving_avg_runs"] = avg
        else:
            match_trends[i]["moving_avg_runs"] = None

    return {
        "batting_history": history_data,
        "intent_by_over": over_intent,
        "match_trends": match_trends,
        "dismissals": dict(dismissal_counts),
        "dismissals_last_5": dict(last_5),
    }

@app.post("/player-bowling-trend-analysis")
def player_bowling_trend_analysis(payload: TrendAnalysisBowlingPayload):
    print("📥 Received Bowling Trend Payload:", payload.dict())
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Resolve player team
    cursor.execute("SELECT country_id FROM players WHERE player_id = ?", (payload.player_id,))
    country_row = cursor.fetchone()
    if not country_row:
        return {"error": "Invalid player_id"}
    country_id = country_row["country_id"]

    cursor.execute("SELECT country_name FROM countries WHERE country_id = ?", (country_id,))
    country_name = cursor.fetchone()["country_name"]

    # Tournament filter
    tournament_ids = []
    tournament_filter = ""
    if payload.tournaments:
        cursor.execute(
            f"SELECT tournament_id FROM tournaments WHERE tournament_name IN ({','.join(['?']*len(payload.tournaments))})",
            payload.tournaments
        )
        tournament_ids = [r["tournament_id"] for r in cursor.fetchall()]
        if tournament_ids:
            tournament_filter = f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    query_args = [payload.player_id, country_name] + tournament_ids

    # Bowler History (with opponent and no econ)
    cursor.execute(f"""
        SELECT 
            m.match_id, 
            m.match_date, 
            t.tournament_name, 
            i.batting_team AS opponent,
            SUM(be.runs + be.wides + be.no_balls) AS runs,
            SUM(
                CASE 
                    WHEN be.dismissal_type IS NOT NULL 
                    AND LOWER(be.dismissal_type) NOT IN ('not out', 'run out', 'obstructing the field', 'retired out', 'handled the ball')
                    AND be.dismissed_player_id = be.batter_id
                    THEN 1 
                    ELSE 0 
                END
            ) AS wickets,

            -- Intent conceded
            ROUND(AVG(be.batting_intent_score), 2) AS intent_conceded

        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id

        WHERE 
            be.bowler_id = ?
            AND i.bowling_team = ?
            {tournament_filter}

        GROUP BY m.match_id
        ORDER BY m.match_date

    """, query_args)

    history = [dict(row) for row in cursor.fetchall()]
    for idx, row in enumerate(history):
        row.update({
            "match_num": idx + 1
        })


    # Consistency Trends
    cursor.execute(f"""
        SELECT m.match_id, m.match_date,
               SUM(be.runs) AS runs,
               COUNT(*) AS balls,
               SUM(be.dot_balls) AS dots,
               SUM(be.wides + be.no_balls) AS extras,
               SUM(CASE WHEN be.pitch_y BETWEEN 0.4 AND 6.0 THEN 1 ELSE 0 END) AS good_length,
               SUM(CASE WHEN (be.edged = 1 OR be.ball_missed = 1) AND be.shot_type != 'Leave' THEN 1 ELSE 0 END) AS false_shots
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.bowler_id = ? AND i.bowling_team = ? {tournament_filter}
        GROUP BY m.match_id
        ORDER BY m.match_date
    """, query_args)

    consistency = []
    for row in cursor.fetchall():
        econ = row["runs"] / (row["balls"] / 6) if row["balls"] else 0
        dot_pct = (row["dots"] / row["balls"] * 100) if row["balls"] else 0
        good_pct = (row["good_length"] / row["balls"] * 100) if row["balls"] else 0
        false_pct = (row["false_shots"] / row["balls"] * 100) if row["balls"] else 0
        consistency.append({
            "match_date": row["match_date"],
            "econ": round(econ, 2),
            "dot_pct": round(dot_pct, 2),
            "good_pct": round(good_pct, 2),
            "extras": row["extras"],
            "false_pct": round(false_pct, 2)
        })
    
    # Step 1: Fetch all dismissals where this bowler was involved
    cursor.execute(f"""
        SELECT LOWER(be.dismissal_type) AS type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.bowler_id = ? AND i.bowling_team = ? {tournament_filter}
        AND be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out'
    """, query_args)

    # Step 2: Filter for dismissals that count towards the bowler
    dismissals = defaultdict(int)
    credited_dismissals = {
        "bowled", "caught", "lbw", "stumped", "hit wicket", "caught and bowled", "hit the ball twice"
    }
    for row in cursor.fetchall():
        dtype = row["type"]
        if dtype in credited_dismissals:
            dismissals[dtype] += 1

    
    print("📤 Returning Bowling Trend Response:", {
        "bowler_history": history,
        "consistency_trends": consistency,
        "dismissal_breakdown": dict(dismissals)
    })

    # Updated SQL query
    cursor.execute(f"""
        SELECT be.pitch_y, bw.bowling_style, be.runs, be.wides, be.no_balls, be.dot_balls,
            be.edged, be.ball_missed, be.shot_type, be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players bw ON be.bowler_id = bw.player_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.bowler_id = ?
        AND i.bowling_team = ?
        AND be.pitch_y IS NOT NULL
        {tournament_filter}
    """, query_args)

    zone_data = cursor.fetchall()

    zone_labels = ["Full Toss", "Yorker", "Full", "Good", "Short"]
    zone_maps = {
        "spin": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        },
        "pace": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        }
    }

    zone_stats = {label: {"balls": 0, "runs": 0, "wickets": 0, "dots": 0, "false_shots": 0} for label in zone_labels}

    for row in zone_data:
        pitch_y = row["pitch_y"]
        style = (row["bowling_style"] or "").lower()
        zone_map = zone_maps["spin"] if "spin" in style else zone_maps["pace"]

        wides = row["wides"] or 0
        no_balls = row["no_balls"] or 0
        legal_delivery = (wides == 0 and no_balls == 0)

        total_runs = (row["runs"] or 0) + wides + no_balls

        for zone, (start, end) in zone_map.items():
            if start <= pitch_y < end:
                if legal_delivery:
                    zone_stats[zone]["balls"] += 1
                    zone_stats[zone]["dots"] += row["dot_balls"] or 0
                zone_stats[zone]["runs"] += total_runs
                if row["dismissal_type"] and row["dismissal_type"].lower() in (
                    "bowled", "caught", "lbw", "stumped", "hit wicket"):
                    zone_stats[zone]["wickets"] += 1
                if legal_delivery and (row["edged"] or row["ball_missed"]) and row["shot_type"] and row["shot_type"].lower() != "leave":
                    zone_stats[zone]["false_shots"] += 1
                break

    # Build final output
    zone_effectiveness = []
    for zone in zone_labels:
        z = zone_stats[zone]
        balls = z["balls"] or 1  # prevent division by zero
        zone_effectiveness.append({
            "zone": zone,
            "balls": z["balls"],
            "runs": z["runs"],
            "wickets": z["wickets"],
            "avg_runs_per_ball": round(z["runs"] / balls, 2),
            "dot_pct": round((z["dots"] / balls) * 100, 2),
            "false_shot_pct": round((z["false_shots"] / balls) * 100, 2)
        })



    return {
        "bowler_history": history,
        "consistency_trends": consistency,
        "dismissal_breakdown": dict(dismissals),
        "zone_effectiveness": sorted(zone_effectiveness, key=lambda x: x["zone"])
    }

@app.post("/match-scorecard")
def get_match_scorecard(payload: MatchScorecardPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    match_id = payload.match_id
    print(f"📅 Requested scorecard for match_id = {match_id}")

    # Match metadata
    cursor.execute("""
        SELECT m.match_date, m.venue, m.toss_winner, 
               t1.country_name AS team1, t2.country_name AS team2
        FROM matches m
        JOIN countries t1 ON m.team_a = t1.country_id
        JOIN countries t2 ON m.team_b = t2.country_id
        WHERE m.match_id = ?
    """, (match_id,))
    match_meta = dict(cursor.fetchone() or {})

    # Get innings for the match
    cursor.execute("""
        SELECT * FROM innings WHERE match_id = ? ORDER BY innings_id ASC
    """, (match_id,))
    innings_list = cursor.fetchall()
    innings_data = []

    for innings in innings_list:
        innings_id = innings["innings_id"]

        # STEP 1: Determine actual batter arrival order (from both batter_id and non_striker_id)
        cursor.execute("""
            SELECT ball_id, batter_id, non_striker_id
            FROM ball_events
            WHERE innings_id = ?
            ORDER BY ball_id ASC
        """, (innings_id,))
        batter_order = []
        seen = set()
        for row in cursor.fetchall():
            for pid in [row["batter_id"], row["non_striker_id"]]:
                if pid and pid not in seen:
                    batter_order.append(pid)
                    seen.add(pid)

        # STEP 2: Get per-batter stats
        cursor.execute("""
            SELECT 
                be.batter_id,
                SUM(be.runs) AS runs,
                COUNT(CASE WHEN be.wides = 0 THEN 1 END) AS balls,
                SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
                SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes
            FROM ball_events be
            WHERE be.innings_id = ?
            GROUP BY be.batter_id
        """, (innings_id,))
        batting_stats = {
            row["batter_id"]: dict(row)
            for row in cursor.fetchall()
        }

        # STEP 3: Get dismissal info from both ball_events and non_ball_dismissals
        dismissal_map = {}

        cursor.execute("""
            SELECT 
                dismissed_player_id,
                dismissal_type,
                fp.player_name AS fielder,
                bp.player_name AS bowler
            FROM ball_events be
            LEFT JOIN players fp ON be.fielder_id = fp.player_id
            LEFT JOIN players bp ON be.bowler_id = bp.player_id
            WHERE be.innings_id = ?
              AND be.dismissal_type IS NOT NULL
        """, (innings_id,))
        for row in cursor.fetchall():
            pid = row["dismissed_player_id"]
            dismissal_map[pid] = {
                "dismissal_type": row["dismissal_type"],
                "fielder": row["fielder"] or "",
                "bowler": row["bowler"] or ""
            }

        # Now apply non_ball_dismissals override if available
        cursor.execute("""
            SELECT player_id, dismissal_type
            FROM non_ball_dismissals
            WHERE innings_id = ?
        """, (innings_id,))
        for row in cursor.fetchall():
            pid = row["player_id"]
            dismissal_map[pid] = {
                "dismissal_type": row["dismissal_type"],
                "fielder": "",
                "bowler": ""
            }

        # STEP 4: Get playing XI for this team
        cursor.execute("SELECT country_id FROM countries WHERE country_name = ?", (innings["batting_team"],))
        batting_team_id = cursor.fetchone()["country_id"]

        cursor.execute("""
            SELECT p.player_id, p.player_name, pmr.is_captain, pmr.is_keeper
            FROM players p
            JOIN player_match_roles pmr ON p.player_id = pmr.player_id
            WHERE pmr.match_id = ? AND pmr.team_id = ?
        """, (match_id, batting_team_id))
        playing_xi = cursor.fetchall()
        role_map = {p["player_id"]: {"is_captain": p["is_captain"], "is_keeper": p["is_keeper"]} for p in playing_xi}
        player_name_map = {p["player_id"]: p["player_name"] for p in playing_xi}

        # STEP 5: Build batting card
        all_seen_ids = set()
        batting_card = []

        for pid in batter_order:
            all_seen_ids.add(pid)
            stats = batting_stats.get(pid, {"runs": 0, "balls": 0, "fours": 0, "sixes": 0})
            dismissal = dismissal_map.get(pid, {})
            dismissal_type = (dismissal.get("dismissal_type") or "").lower()
            fielder_text = ""
            bowler_text = ""

            if dismissal_type:
                bowler = dismissal.get("bowler", "")
                fielder = dismissal.get("fielder", "")
                if dismissal_type in ["bowled", "lbw"]:
                    bowler_text = f"b. {bowler}"
                elif dismissal_type == "caught":
                    fielder_text = f"c. {fielder}"
                    bowler_text = f"b. {bowler}"
                elif dismissal_type == "run out":
                    fielder_text = f"({fielder})"
                    bowler_text = "run out"
                elif dismissal_type == "stumped":
                    fielder_text = f"st. {fielder}"
                    bowler_text = f"b. {bowler}"
                else:
                    bowler_text = f"{dismissal_type.title()}. {bowler}"

            batting_card.append({
                "player_id": pid,
                "player": player_name_map.get(pid, "Unknown"),
                "runs": stats["runs"],
                "balls": stats["balls"],
                "fours": stats["fours"],
                "sixes": stats["sixes"],
                "strike_rate": round((stats["runs"] / stats["balls"]) * 100, 2) if stats["balls"] else 0,
                "fielder_text": fielder_text,
                "bowler_text": bowler_text,
                "is_captain": role_map.get(pid, {}).get("is_captain", 0),
                "is_keeper": role_map.get(pid, {}).get("is_keeper", 0)
            })

        # STEP 6: Add "Did Not Bat" players
        for player in playing_xi:
            pid = player["player_id"]
            if pid not in all_seen_ids:
                batting_card.append({
                    "player_id": pid,
                    "player": player["player_name"],
                    "runs": "-",
                    "balls": "-",
                    "fours": "-",
                    "sixes": "-",
                    "strike_rate": "-",
                    "fielder_text": "Did Not Bat",
                    "bowler_text": "",
                    "is_captain": player["is_captain"],
                    "is_keeper": player["is_keeper"]
                })


        # Bowling Card ordered by appearance
        cursor.execute("""
            SELECT 
                be.bowler_id,
                p.player_name,
                MIN(be.ball_id) AS first_ball_id,
                SUM(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END) AS legal_balls,
                SUM(CASE WHEN be.runs = 0 AND be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END) AS dots,
                SUM(be.runs + IFNULL(be.wides, 0) + IFNULL(be.no_balls, 0)) AS runs,
                SUM(CASE 
                    WHEN be.dismissed_player_id = be.batter_id
                     AND LOWER(be.dismissal_type) NOT IN ('not out', 'run out', 'retired hurt', 'retired out')
                    THEN 1 ELSE 0 END) AS wickets,
                SUM(be.wides) AS wides,
                SUM(be.no_balls) AS no_balls
            FROM ball_events be
            JOIN players p ON be.bowler_id = p.player_id
            WHERE be.innings_id = ?
            GROUP BY be.bowler_id
            ORDER BY first_ball_id ASC
        """, (innings_id,))

        bowling_card = []
        for row in cursor.fetchall():
            legal_balls = row["legal_balls"]
            overs = f"{legal_balls // 6}.{legal_balls % 6}"
            economy = round(row["runs"] / (legal_balls / 6), 2) if legal_balls else 0

            bowling_card.append({
                "player_id": row["bowler_id"],
                "bowler": row["player_name"],
                "overs": overs,
                "dots": row["dots"],
                "runs": row["runs"],
                "wickets": row["wickets"],
                "wides": row["wides"],
                "no_balls": row["no_balls"],
                "economy": economy
            })

        # Fall of Wickets
        cursor.execute("""
            SELECT 
                be.ball_id,
                be.over_number,
                be.balls_this_over,
                be.dismissed_player_id,
                p.player_name AS dismissed_name,
                SUM(be2.runs + IFNULL(be2.wides, 0) + IFNULL(be2.no_balls, 0) + 
                    IFNULL(be2.byes, 0) + IFNULL(be2.leg_byes, 0) + IFNULL(be2.penalty_runs, 0)) AS cumulative_score
            FROM ball_events be
            JOIN players p ON be.dismissed_player_id = p.player_id
            JOIN ball_events be2 ON be2.innings_id = be.innings_id AND be2.ball_id <= be.ball_id
            WHERE be.innings_id = ?
            AND be.dismissal_type IS NOT NULL
            AND LOWER(be.dismissal_type) != 'not out'
            GROUP BY be.ball_id
            ORDER BY be.ball_id
        """, (innings_id,))

        fall_of_wickets = []
        for i, row in enumerate(cursor.fetchall()):
            over = int(row['over_number'])
            ball = int(row['balls_this_over'])
            over_notation = f"{over}.{ball}"

            fall_of_wickets.append(
                f"{row['cumulative_score']}/{i+1} ({row['dismissed_name']} - {over_notation} ov)"
            )


        # Extras
        cursor.execute("""
            SELECT SUM(wides) AS wides,
                   SUM(no_balls) AS no_balls,
                   SUM(byes) AS byes,
                   SUM(leg_byes) AS leg_byes,
                   SUM(penalty_runs) AS penalty
            FROM ball_events
            WHERE innings_id = ?
        """, (innings_id,))
        extras = dict(cursor.fetchone() or {})
        extras["total"] = sum(v or 0 for v in extras.values())

        # Total
        cursor.execute("""
            SELECT SUM(runs) + 
                   SUM(wides) + 
                   SUM(no_balls) + 
                   SUM(byes) + 
                   SUM(leg_byes) + 
                   SUM(penalty_runs) AS total_runs
            FROM ball_events
            WHERE innings_id = ?
        """, (innings_id,))
        total_score = cursor.fetchone()["total_runs"]

        overs = innings["overs_bowled"]
        formatted_overs = f"{int(overs)}.{int((overs % 1) * 6)}"

        innings_data.append({
            "team": innings["batting_team"],
            "batting_card": batting_card,
            "bowling_card": bowling_card,
            "fall_of_wickets": fall_of_wickets,
            "extras": extras,
            "total": total_score or 0,
            "overs": formatted_overs
        })

    cursor.execute("SELECT result FROM matches WHERE match_id = ?", (match_id,))
    result = cursor.fetchone()
    result_text = result["result"] if result else "Result not available"

    conn.close()
    return {
        "meta": match_meta,
        "innings": innings_data,
        "result": result_text
    }

@app.get("/match-list")
def get_matches(team_category: str, tournament: str):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if team_category.lower() == "training":
        cursor.execute("""
            SELECT m.match_id, m.match_date, t1.country_name AS team1, t2.country_name AS team2
            FROM matches m
            JOIN countries t1 ON m.team1_id = t1.country_id
            JOIN countries t2 ON m.team2_id = t2.country_id
            JOIN tournaments t ON m.tournament_id = t.tournament_id
            WHERE t.tournament_name = ? AND (LOWER(t1.country_name) LIKE ? OR LOWER(t2.country_name) LIKE ?)
            ORDER BY m.match_date DESC
        """, (tournament, "%training%", "%training%"))
    else:
        cursor.execute("""
            SELECT m.match_id, m.match_date, t1.country_name AS team1, t2.country_name AS team2
            FROM matches m
            JOIN countries t1 ON m.team1_id = t1.country_id
            JOIN countries t2 ON m.team2_id = t2.country_id
            JOIN tournaments t ON m.tournament_id = t.tournament_id
            WHERE t.tournament_name = ? AND (
                (t1.country_name LIKE ? AND LOWER(t1.country_name) NOT LIKE ?) OR
                (t2.country_name LIKE ? AND LOWER(t2.country_name) NOT LIKE ?)
            )
            ORDER BY m.match_date DESC
        """, (tournament, f"%{team_category}", "%training%", f"%{team_category}", "%training%"))

    matches = cursor.fetchall()
    return [f"{row['match_date']} - {row['team1']} vs {row['team2']} (ID: {row['match_id']})" for row in matches]

@app.post("/match-momentum")
def get_match_momentum(payload: MatchPressurePayload):
    import sqlite3
    match_id = payload.match_id
    print(f"📥 Generating momentum for match_id = {match_id}")

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Just pull innings info directly with team name
    cursor.execute("""
        SELECT innings_id, innings, batting_team
        FROM innings
        WHERE match_id = ?
        ORDER BY innings ASC
    """, (match_id,))
    innings_list = cursor.fetchall()

    print(f"🧩 Found innings: {[dict(row) for row in innings_list]}")

    result = []

    for row in innings_list:
        innings_id = row["innings_id"]
        team_name = row["batting_team"]

        cursor.execute("""
            SELECT 
              CAST(over_number AS INT) AS over,
              AVG(batting_bpi) AS avg_batting_bpi,
              AVG(bowling_bpi) AS avg_bowling_bpi,
              SUM(CASE WHEN dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS wickets
            FROM ball_events
            WHERE innings_id = ?
            GROUP BY CAST(over_number AS INT)
            ORDER BY over
        """, (innings_id,))
        overs = cursor.fetchall()

        momentum_data = []
        for over_row in overs:
            over = over_row["over"]
            batting = over_row["avg_batting_bpi"] or 0
            bowling = over_row["avg_bowling_bpi"] or 0
            net = round(bowling - batting, 2)
            momentum_data.append({
                "over": over,
                "batting_bpi": round(batting, 2),
                "bowling_bpi": round(bowling, 2),
                "net_momentum": net,
                "wickets": over_row["wickets"] or 0
            })

        result.append({
            "team": team_name,
            "momentum": momentum_data
        })

    print("✅ Returning momentum result:", result)
    conn.close()
    return {"momentum": result}

@app.post("/match-partnerships")
def get_match_partnerships(payload: MatchPartnershipsPayload):
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get partnerships for the innings in the match
    cursor.execute("""
        SELECT p.partnership_id, p.innings_id, p.start_wicket, p.batter1_id, p.batter2_id,
               p.start_over, p.end_over, p1.player_name AS batter1_name, p2.player_name AS batter2_name,
               i.batting_team AS batting_team_name
        FROM partnerships p
        LEFT JOIN players p1 ON p.batter1_id = p1.player_id
        LEFT JOIN players p2 ON p.batter2_id = p2.player_id
        JOIN innings i ON p.innings_id = i.innings_id
        WHERE p.innings_id IN (SELECT innings_id FROM innings WHERE match_id = ?)
        ORDER BY p.start_wicket ASC
    """, (payload.match_id,))
    partnership_rows = cursor.fetchall()

    partnerships = []

    for p in partnership_rows:
        # Use the tested SQL directly
        cursor.execute("""
            WITH balls_in_partnership AS (
                SELECT 
                    over_number,
                    ball_number,
                    batter_id,
                    non_striker_id,
                    runs,
                    wides,
                    no_balls,
                    byes,
                    leg_byes
                FROM ball_events
                WHERE innings_id = ?
                  AND (
                    (batter_id = ? AND non_striker_id = ?)
                    OR (batter_id = ? AND non_striker_id = ?)
                  )
            )
            SELECT
                (SELECT MIN(over_number || '.' || ball_number) FROM balls_in_partnership) AS start_ball,
                (SELECT MAX(over_number || '.' || ball_number) FROM balls_in_partnership) AS end_ball,
                (SELECT SUM(runs + wides + no_balls + byes + leg_byes) FROM balls_in_partnership) AS partnership_runs,
                (SELECT COUNT(*) FROM balls_in_partnership WHERE wides = 0) AS partnership_legal_balls,
                (SELECT SUM(runs) FROM balls_in_partnership WHERE batter_id = ?) AS batter1_runs,
                (SELECT COUNT(*) FROM balls_in_partnership WHERE batter_id = ? AND wides = 0) AS batter1_legal_balls,
                (SELECT SUM(runs) FROM balls_in_partnership WHERE batter_id = ?) AS batter2_runs,
                (SELECT COUNT(*) FROM balls_in_partnership WHERE batter_id = ? AND wides = 0) AS batter2_legal_balls
        """, (
            p["innings_id"],
            p["batter1_id"], p["batter2_id"],
            p["batter2_id"], p["batter1_id"],
            p["batter1_id"],
            p["batter1_id"],
            p["batter2_id"],
            p["batter2_id"]
        ))

        stats = cursor.fetchone()

        partnerships.append({
            "partnership_id": p["partnership_id"],
            "innings_id": p["innings_id"],
            "batting_team": p["batting_team_name"],
            "start_wicket": p["start_wicket"],
            "batter1_name": p["batter1_name"],
            "batter2_name": p["batter2_name"],
            "start_ball": stats["start_ball"],
            "end_ball": stats["end_ball"],
            "partnership_runs": stats["partnership_runs"] or 0,
            "partnership_legal_balls": stats["partnership_legal_balls"] or 0,
            "batter1_runs": stats["batter1_runs"] or 0,
            "batter1_legal_balls": stats["batter1_legal_balls"] or 0,
            "batter2_runs": stats["batter2_runs"] or 0,
            "batter2_legal_balls": stats["batter2_legal_balls"] or 0,
        })

    conn.close()
    return {"partnerships": partnerships}

@app.get("/partnership-details/{partnership_id}")
def get_partnership_details(partnership_id: int):
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get partnership details
    cursor.execute("""
        SELECT 
            p.innings_id,
            p.batter1_id,
            p.batter2_id
        FROM partnerships p
        WHERE p.partnership_id = ?
    """, (partnership_id,))
    p_row = cursor.fetchone()

    if not p_row:
        conn.close()
        return {}

    innings_id = p_row["innings_id"]
    batter1_id = p_row["batter1_id"]
    batter2_id = p_row["batter2_id"]

    # Get all balls in this partnership
    cursor.execute("""
        SELECT 
            runs,
            wides,
            no_balls,
            byes,
            leg_byes,
            batter_id,
            shot_x,
            shot_y,
            batting_intent_score,
            dismissal_type
        FROM ball_events
        WHERE innings_id = ?
          AND (
              (batter_id = ? AND non_striker_id = ?)
           OR (batter_id = ? AND non_striker_id = ?)
          )
    """, (innings_id, batter1_id, batter2_id, batter2_id, batter1_id))

    balls = cursor.fetchall()

    # Calculate metrics
    total_runs = 0
    total_balls = 0
    total_intent = 0
    intent_count = 0
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    sixes = 0
    extras = 0
    scoring_shots = 0

    wagon_wheel_data = []

    for b in balls:
        runs = b["runs"] or 0
        wides = b["wides"] or 0
        no_balls = b["no_balls"] or 0
        byes = b["byes"] or 0
        leg_byes = b["leg_byes"] or 0

        extras += wides + no_balls + byes + leg_byes
        total_runs += runs + wides + no_balls + byes + leg_byes

        if wides == 0:
            total_balls += 1
            if runs > 0:
                scoring_shots += 1

        if b["batting_intent_score"] is not None:
            total_intent += b["batting_intent_score"]
            intent_count += 1

        if runs == 1:
            ones += 1
        elif runs == 2:
            twos += 1
        elif runs == 3:
            threes += 1
        elif runs == 4:
            fours += 1
        elif runs == 6:
            sixes += 1

        # Wagon wheel data with dismissal_type included for frontend filter logic
        if b["shot_x"] is not None and b["shot_y"] is not None:
            wagon_wheel_data.append({
                "x": b["shot_x"],
                "y": b["shot_y"],
                "runs": runs,
                "dismissal_type": b["dismissal_type"]  # Add this field for frontend compatibility
            })

    average_intent = round(total_intent / intent_count, 2) if intent_count > 0 else 0
    scoring_shot_pct = round((scoring_shots / total_balls) * 100, 2) if total_balls > 0 else 0

    conn.close()

    return {
        "summary": {
            "total_runs": total_runs,
            "total_balls": total_balls,
            "average_intent": average_intent,
            "ones": ones,
            "twos": twos,
            "threes": threes,
            "fours": fours,
            "sixes": sixes,
            "extras": extras,
            "scoring_shot_pct": scoring_shot_pct
        },
        "wagon_wheel": wagon_wheel_data
    }

@app.post("/player-detailed-batting")
def get_player_detailed_batting(payload: PlayerDetailedBattingPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get tournament_ids
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row["tournament_id"] for row in cursor.fetchall()]
    
    if not tournament_ids:
        conn.close()
        return {"pitch_map": [], "wagon_wheel": [], "full_balls": []}

    filters = list(payload.player_ids) + tournament_ids

    # Filters
    batter_filter = f"AND be.batter_id IN ({','.join(['?'] * len(payload.player_ids))})"
    tournament_filter = f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    match_filter = ""
    if payload.match_id:
        match_filter = "AND m.match_id = ?"
        filters.append(payload.match_id)

    bowling_arm_filter = ""
    if payload.bowling_arm:
        placeholders = ",".join(["?"] * len(payload.bowling_arm))
        bowling_arm_filter = f"AND p.bowling_arm IN ({placeholders})"
        filters.extend(payload.bowling_arm)

    bowling_style_filter = ""
    if payload.bowling_style:
        placeholders = ",".join(["?"] * len(payload.bowling_style))
        bowling_style_filter = f"AND p.bowling_style IN ({placeholders})"
        filters.extend(payload.bowling_style)

    length_filter = ""
    if payload.lengths:
        conditions = []
        for length in payload.lengths:
            if length == "Full Toss":
                conditions.append("(be.pitch_y BETWEEN -0.090909 AND 0.036363636)")
            elif length == "Yorker":
                conditions.append("(be.pitch_y BETWEEN 0.036363636 AND 0.1636363636)")
            elif length == "Full":
                conditions.append("(be.pitch_y BETWEEN 0.1636363636 AND 0.318181818)")
            elif length == "Good":
                conditions.append("(be.pitch_y BETWEEN 0.318181818 AND 0.5454545454)")
            elif length == "Short":
                conditions.append("(be.pitch_y BETWEEN 0.5454545454 AND 1.0)")
        if conditions:
            length_filter = "AND (" + " OR ".join(conditions) + ")"

    # PITCH MAP + FULL BALL DATA
    cursor.execute(f"""
        SELECT 
            be.pitch_x,
            be.pitch_y,
            be.runs,
            be.wides,
            be.no_balls,       
            be.ball_id,
            CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END AS wicket,
            be.dismissal_type,
            p.player_name AS bowler_name,
            p.bowling_style AS bowler_type,
            p.bowling_arm,
            be.delivery_type,
            be.over_number,
            be.balls_this_over,
            be.shot_type,
            be.footwork,
            be.shot_selection
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.bowler_id = p.player_id
        WHERE 1=1
        {batter_filter}
        {tournament_filter}
        {match_filter}
        {bowling_arm_filter}
        {bowling_style_filter}
        {length_filter}
        AND be.pitch_x IS NOT NULL
        AND be.pitch_y IS NOT NULL
    """, filters)

    pitch_map = []
    full_balls = []
    for row in cursor.fetchall():
        pitch_map.append({
            "pitch_x": row["pitch_x"],
            "pitch_y": row["pitch_y"],
            "runs": row["runs"],
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "ball_id": row["ball_id"],
            "wicket": bool(row["wicket"]),
            "dismissal_type": row["dismissal_type"]
        })

        full_balls.append({
            "pitch_x": row["pitch_x"],
            "pitch_y": row["pitch_y"],
            "runs": row["runs"],
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "ball_id": row["ball_id"],
            "wicket": bool(row["wicket"]),
            "dismissal_type": row["dismissal_type"],
            "bowler_name": row["bowler_name"],
            "bowler_type": row["bowler_type"],
            "delivery_type": row["delivery_type"],
            "bowling_arm": row["bowling_arm"],
            "over": row["over_number"],
            "balls_this_over": row["balls_this_over"],
            "shot_type": row["shot_type"],
            "footwork": row["footwork"],
            "shot_selection": row["shot_selection"]
        })

    # WAGON WHEEL QUERY
    cursor.execute(f"""
        SELECT 
            be.shot_x,
            be.shot_y,
            be.runs,
            be.over_number,
            be.balls_this_over,
            be.dismissal_type,
            be.ball_id
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.bowler_id = p.player_id
        WHERE 1=1
        {batter_filter}
        {tournament_filter}
        {match_filter}
        {bowling_arm_filter}
        {bowling_style_filter}
        {length_filter}
        AND be.shot_x IS NOT NULL
        AND be.shot_y IS NOT NULL
    """, filters)

    wagon_wheel = []
    for row in cursor.fetchall():
        wagon_wheel.append({
            "shot_x": row["shot_x"],
            "shot_y": row["shot_y"],
            "runs": row["runs"],
            "over": row["over_number"],
            "balls_this_over": row["balls_this_over"],
            "dismissal_type": row["dismissal_type"],
            "ball_id": row["ball_id"]
        })

    conn.close()

    return {
        "pitch_map": pitch_map,
        "wagon_wheel": wagon_wheel,
        "full_balls": full_balls
    }

@app.post("/player-intent-summary")
def get_player_intent_summary(payload: PlayerIntentSummaryPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get tournament_ids
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row["tournament_id"] for row in cursor.fetchall()]
    if not tournament_ids:
        return {}

    filters = list(payload.player_ids) + tournament_ids
    batter_filter = f"AND be.batter_id IN ({','.join(['?'] * len(payload.player_ids))})"
    tournament_filter = f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    match_filter = ""
    if payload.match_id:
        match_filter = "AND m.match_id = ?"
        filters.append(payload.match_id)

    bowling_arm_filter = ""
    if payload.bowling_arm:
        placeholders = ",".join(["?"] * len(payload.bowling_arm))
        bowling_arm_filter = f"AND bp.bowling_arm IN ({placeholders})"
        filters.extend(payload.bowling_arm)

    bowling_style_filter = ""
    if payload.bowling_style:
        placeholders = ",".join(["?"] * len(payload.bowling_style))
        bowling_style_filter = f"AND bp.bowling_style IN ({placeholders})"
        filters.extend(payload.bowling_style)

    length_filter = ""
    if payload.lengths:
        conditions = []
        for length in payload.lengths:
            if length == "Full Toss":
                conditions.append("(be.pitch_y BETWEEN -0.090909 AND 0.036363636)")
            elif length == "Yorker":
                conditions.append("(be.pitch_y BETWEEN 0.036363636 AND 0.1636363636)")
            elif length == "Full":
                conditions.append("(be.pitch_y BETWEEN 0.1636363636 AND 0.318181818)")
            elif length == "Good":
                conditions.append("(be.pitch_y BETWEEN 0.318181818 AND 0.5454545454)")
            elif length == "Short":
                conditions.append("(be.pitch_y BETWEEN 0.5454545454 AND 1.0)")
        if conditions:
            length_filter = "AND (" + " OR ".join(conditions) + ")"

    query = f"""
        SELECT be.ball_id, be.runs, be.batting_intent_score AS intent, be.dismissal_type, be.shot_selection
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players bp ON be.bowler_id = bp.player_id
        WHERE 1=1
        {batter_filter}
        {tournament_filter}
        {match_filter}
        {bowling_arm_filter}
        {bowling_style_filter}
        {length_filter}
        ORDER BY be.ball_id
    """

    cursor.execute(query, filters)
    rows = cursor.fetchall()
    conn.close()

    # Aggregate
    total_runs = 0
    total_intent = 0
    total_balls = 0
    scoring_shots = 0

    intent_after_dot = 0
    dot_followup_balls = 0
    scoring_after_dot = 0
    prev_run = None
    dismissal_counter = Counter()
    dismissal_after_dot_counter = Counter()
    shot_selection_counter = Counter()

    for row in rows:
        intent = row["intent"]
        runs = row["runs"]
        dismissal_type = row["dismissal_type"]
        if dismissal_type and dismissal_type.lower() not in ["", "not out"]:
            label = dismissal_type.strip().title()
            dismissal_counter[label] += 1

            if prev_run == 0:
                dismissal_after_dot_counter[label] += 1

        if row["shot_selection"]:
            shot = row["shot_selection"].strip().title()
            shot_selection_counter[shot] += 1

        if intent is not None:
            total_intent += intent
            total_balls += 1
            total_runs += runs or 0
            if runs > 0:
                scoring_shots += 1

            if prev_run == 0:
                intent_after_dot += intent
                dot_followup_balls += 1
                if runs > 0:
                    scoring_after_dot += 1

        prev_run = runs

    return {
        "total_runs": total_runs,
        "balls_faced": total_balls,
        "scoring_shot_pct": round((scoring_shots / total_balls) * 100, 2) if total_balls else 0,
        "average_intent": round(total_intent / total_balls, 2) if total_balls else 0,
        "scoring_shot_pct_after_dot": round((scoring_after_dot / dot_followup_balls) * 100, 2) if dot_followup_balls else 0,
        "average_intent_after_dot": round(intent_after_dot / dot_followup_balls, 2) if dot_followup_balls else 0,
        "dot_followups": dot_followup_balls,
        "dismissals": dict(dismissal_counter),
        "dismissals_after_dot": dict(dismissal_after_dot_counter),
        "shot_selection": dict(shot_selection_counter)
    }

@app.post("/player-detailed-bowling")
def get_player_detailed_bowling(payload: PlayerDetailedBowlingPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get tournament_ids
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row["tournament_id"] for row in cursor.fetchall()]

    if not tournament_ids:
        conn.close()
        return {"pitch_map": [], "wagon_wheel": [], "full_balls": []}

    filters = list(payload.player_ids) + tournament_ids

    bowler_filter = f"AND be.bowler_id IN ({','.join(['?'] * len(payload.player_ids))})"
    tournament_filter = f"AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    match_filter = ""
    if payload.match_id:
        match_filter = "AND m.match_id = ?"
        filters.append(payload.match_id)

    batting_hand_filter = ""
    if payload.batting_hand:
        placeholders = ",".join(["?"] * len(payload.batting_hand))
        batting_hand_filter = f"AND batter.batting_hand IN ({placeholders})"
        filters.extend(payload.batting_hand)

    bowling_style_filter = ""
    if payload.bowling_style:
        placeholders = ",".join(["?"] * len(payload.bowling_style))
        bowling_style_filter = f"AND p.bowling_style IN ({placeholders})"
        filters.extend(payload.bowling_style)

    length_filter = ""
    if payload.lengths:
        conditions = []
        for length in payload.lengths:
            if length == "Full Toss":
                conditions.append("(be.pitch_y BETWEEN -0.090909 AND 0.036363636)")
            elif length == "Yorker":
                conditions.append("(be.pitch_y BETWEEN 0.036363636 AND 0.1636363636)")
            elif length == "Full":
                conditions.append("(be.pitch_y BETWEEN 0.1636363636 AND 0.318181818)")
            elif length == "Good":
                conditions.append("(be.pitch_y BETWEEN 0.318181818 AND 0.5454545454)")
            elif length == "Short":
                conditions.append("(be.pitch_y BETWEEN 0.5454545454 AND 1.0)")
        if conditions:
            length_filter = "AND (" + " OR ".join(conditions) + ")"

    # PITCH MAP + FULL BALL DATA
    cursor.execute(f"""
        SELECT 
            be.pitch_x,
            be.pitch_y,
            be.runs,
            be.wides,
            be.no_balls,       
            be.ball_id,
            CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END AS wicket,
            be.dismissal_type,
            batter.player_name AS batter_name,
            batter.batting_hand,
            p.player_name AS bowler_name,
            p.bowling_arm,
            p.bowling_style AS bowler_type,
            be.delivery_type,
            be.over_number,
            be.balls_this_over,
            be.shot_type,
            be.footwork,
            be.shot_selection
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.bowler_id = p.player_id
        JOIN players batter ON be.batter_id = batter.player_id
        WHERE 1=1
        {bowler_filter}
        {tournament_filter}
        {match_filter}
        {batting_hand_filter}
        {bowling_style_filter}
        {length_filter}
        AND be.pitch_x IS NOT NULL
        AND be.pitch_y IS NOT NULL
    """, filters)

    pitch_map = []
    full_balls = []
    for row in cursor.fetchall():
        pitch_map.append({
            "pitch_x": row["pitch_x"],
            "pitch_y": row["pitch_y"],
            "runs": row["runs"],
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "ball_id": row["ball_id"],
            "wicket": bool(row["wicket"]),
            "dismissal_type": row["dismissal_type"]
        })

        full_balls.append({
            "pitch_x": row["pitch_x"],
            "pitch_y": row["pitch_y"],
            "runs": row["runs"],
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "ball_id": row["ball_id"],
            "wicket": bool(row["wicket"]),
            "dismissal_type": row["dismissal_type"],
            "batter_name": row["batter_name"],
            "batting_hand": row["batting_hand"],
            "bowler_name": row["bowler_name"],
            "bowling_arm": row["bowling_arm"],
            "bowler_type": row["bowler_type"],
            "delivery_type": row["delivery_type"],
            "over": row["over_number"],
            "balls_this_over": row["balls_this_over"],
            "shot_type": row["shot_type"],
            "footwork": row["footwork"],
            "shot_selection": row["shot_selection"]
        })


    # WAGON WHEEL — runs conceded by bowler
    cursor.execute(f"""
        SELECT 
            be.shot_x,
            be.shot_y,
            be.runs,
            be.over_number,
            be.balls_this_over,
            be.dismissal_type,
            be.ball_id
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.bowler_id = p.player_id
        JOIN players batter ON be.batter_id = batter.player_id
        WHERE 1=1
        {bowler_filter}
        {tournament_filter}
        {match_filter}
        {batting_hand_filter}
        {bowling_style_filter}
        {length_filter}
        AND be.shot_x IS NOT NULL
        AND be.shot_y IS NOT NULL
    """, filters)

    wagon_wheel = []
    for row in cursor.fetchall():
        wagon_wheel.append({
            "shot_x": row["shot_x"],
            "shot_y": row["shot_y"],
            "runs": row["runs"],
            "over": row["over_number"],
            "balls_this_over": row["balls_this_over"],
            "dismissal_type": row["dismissal_type"],
            "ball_id": row["ball_id"]
        })

    conn.close()

    return {
        "pitch_map": pitch_map,
        "wagon_wheel": wagon_wheel,
        "full_balls": full_balls
    }

@app.get("/match-report/{match_id}/player/{player_id}")
def match_report(match_id: int, player_id: int):
    data = fetch_player_match_stats(match_id, player_id)
    if not data:
        raise HTTPException(status_code=404, detail="Player or match not found")

    pdf_buffer = generate_pdf_report(data)
    headers = {
        'Content-Disposition': f'inline; filename="match_report_player_{player_id}_match_{match_id}.pdf"'
    }
    return StreamingResponse(pdf_buffer, media_type='application/pdf', headers=headers)

@app.get("/team-match-report/{match_id}/{team_id}/pdf")
def team_match_report_pdf(match_id: int, team_id: int):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Match Summary & Top 3s
    match_summary = fetch_match_summary(cursor, match_id, team_id)

    # Extract the batting team's name
    team_name = match_summary["team_a"] if "Brasil" in match_summary["team_a"] else match_summary["team_b"]

    # KPIs & Medal Tally
    kpis, medal_tally = calculate_kpis(cursor, match_id, team_id, team_name)

    over_medals = calculate_over_medals(cursor, match_id, team_name)

    # Generate PDF
    pdf_data = {
        "match_summary": match_summary,
        "kpis": kpis,
        "medal_tallies_by_area": medal_tally,
        "over_medals": over_medals
    }
    pdf = generate_team_pdf_report(pdf_data)

    return StreamingResponse(pdf, media_type="application/pdf")

@app.post("/match-ball-by-ball")
def get_match_ball_by_ball(payload: MatchBallByBallPayload):
    import os
    import sqlite3

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Validate match ID
    cursor.execute("""
        SELECT match_id FROM matches WHERE match_id = ?
    """, (payload.match_id,))
    if not cursor.fetchone():
        conn.close()
        return {"error": "Match not found."}

    # Main data query with batting team name
    cursor.execute("""
        SELECT 
            be.innings_id,
            be.over_number,
            be.ball_number,
            be.batter_id,
            be.bowler_id,
            be.non_striker_id,
            be.runs,
            be.wides,
            be.no_balls,
            be.byes,
            be.leg_byes,
            be.dismissal_type,
            p1.player_name AS bowler_name,
            i.batting_team AS batting_team
        FROM ball_events be
        JOIN players p1 ON be.bowler_id = p1.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ?
        ORDER BY be.innings_id, CAST(be.over_number AS REAL), be.ball_number
    """, (payload.match_id,))

    balls = []
    for row in cursor.fetchall():
        outcome = ""
        if row["wides"]:
            outcome = f"[Wd{row['wides'] if row['wides'] > 1 else ''}]"
        elif row["no_balls"]:
            outcome = f"[NB{row['runs'] if row['runs'] else ''}]"
            if row["byes"]:
                outcome += f"+{row['byes']}B"
            elif row["leg_byes"]:
                outcome += f"+{row['leg_byes']}LB"
        elif row["byes"]:
            outcome = f"[{row['byes']}B]"
        elif row["leg_byes"]:
            outcome = f"[{row['leg_byes']}LB]"
        elif row["dismissal_type"] and row["dismissal_type"] != "not out":
            outcome = "W"
        else:
            outcome = str(row["runs"])

        balls.append({
            "innings_id": row["innings_id"],
            "over_number": row["over_number"],
            "ball_number": row["ball_number"],
            "bowler_name": row["bowler_name"],
            "batting_team": row["batting_team"],  # ✅ Include batting team
            "runs": row["runs"] or 0,
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "byes": row["byes"] or 0,
            "leg_byes": row["leg_byes"] or 0,
            "dismissal_type": row["dismissal_type"],
            "outcome": outcome
        })

    conn.close()
    return {"balls": balls}

@app.post("/api/upload-wagon-wheel")
async def upload_wagon_wheel(request: Request):
    data = await request.json()
    base64_image = data["image"]
    image_type = data.get("type", "wagon_wheel")  # e.g., "wagon_wheel" or "pitch_map"

    # Remove the data URL header
    header, encoded = base64_image.split(",", 1)
    image_data = base64.b64decode(encoded)

    # Save to a temp location
    filename = f"/tmp/{image_type}_chart.png"
    with open(filename, "wb") as f:
        f.write(image_data)

    return {"message": f"{image_type} image saved successfully"}

@app.post("/api/upload-pitch-map")
async def upload_pitch_map(request: Request):
    data = await request.json()
    base64_image = data["image"]
    header, encoded = base64_image.split(",", 1)
    image_data = base64.b64decode(encoded)
     
    filename = "/tmp/pitch_map_chart.png"
    with open(filename, "wb") as f:
        f.write(image_data)
    
    return {"message": "pitch_map image saved successfully"}

@app.get("/player-wagon-wheel-data")
def player_wagon_wheel_data(matchId: int, playerId: int):
    conn = sqlite3.connect("cricket_analysis.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT be.shot_x, be.shot_y, be.runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ? AND be.shot_x IS NOT NULL AND be.shot_y IS NOT NULL
    """, (matchId, playerId))

    shots = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return shots

@app.get("/player-pitch-map-data")
def player_pitch_map_data(matchId: int, playerId: int):
    conn = sqlite3.connect("cricket_analysis.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Use a more sophisticated query to get pitch map data just for this player/match
    cursor.execute("""
        SELECT be.pitch_x, be.pitch_y, be.runs, be.wides, be.no_balls, be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.bowler_id = ? AND be.pitch_x IS NOT NULL AND be.pitch_y IS NOT NULL
    """, (matchId, playerId))

    data = [
        {
            "pitch_x": row["pitch_x"],
            "pitch_y": row["pitch_y"],
            "runs": row["runs"] or 0,
            "wides": row["wides"] or 0,
            "no_balls": row["no_balls"] or 0,
            "dismissal_type": row["dismissal_type"]
        }
        for row in cursor.fetchall()
    ]


    conn.close()
    return data

@app.post("/tactical-matchup-detailed")
def get_tactical_matchup_detail(payload: MatchupDetailPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT player_name FROM players WHERE player_id = ?", (payload.player_id,))
    player = cursor.fetchone()
    if not player:
        conn.close()
        return {}

    batter_name = player["player_name"]

    effectiveness = {}
    detailed_stats = {}

    # 🟩 Query for each bowler style
    for style in ["Pace", "Medium", "Off Spin", "Leg Spin"]:
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END) AS legal_balls,
                SUM(be.runs) AS runs,
                SUM(CASE WHEN be.runs=0 AND be.wides=0 THEN 1 ELSE 0 END) AS dots,
                SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS outs
            FROM ball_events be
            JOIN players bowl ON be.bowler_id = bowl.player_id
            WHERE be.batter_id = ? AND LOWER(bowl.bowling_style) = LOWER(?)
        """, (payload.player_id, style))
        row = cursor.fetchone()

        # Skip styles with no data
        if not row or not row["legal_balls"]:
            detailed_stats[style] = {"balls": 0, "rpb": 0, "dot_pct": 0, "dismissal_pct": 0}
            effectiveness[style] = 0
            continue

        balls = row["legal_balls"]
        runs = row["runs"] or 0
        outs = row["outs"] or 0
        rpb = round(runs / balls, 2) if balls else 0
        rpb_safe = max(rpb, 0.1)
        dot_pct = round((row["dots"] or 0) * 100 / balls, 1)
        out_pct = round(outs * 100 / balls, 1)

        effectiveness_score = (out_pct / 100) + (1 / rpb_safe)
        effectiveness[style] = effectiveness_score

        detailed_stats[style] = {
            "balls": balls,
            "rpb": rpb,
            "dot_pct": dot_pct,
            "dismissal_pct": out_pct
        }

    # 🔎 Determine best bowler type
    recommended_type = max(effectiveness, key=effectiveness.get)

    # 🟩 Retrieve all balls for zone analysis (best bowler type only)
    cursor.execute("""
        SELECT be.pitch_x, be.pitch_y, be.runs, be.dismissal_type
        FROM ball_events be
        JOIN players bowl ON be.bowler_id = bowl.player_id
        WHERE be.batter_id = ? AND LOWER(bowl.bowling_style) = LOWER(?)
          AND be.wides = 0 AND be.pitch_x IS NOT NULL AND be.pitch_y IS NOT NULL
    """, (payload.player_id, recommended_type))
    balls = cursor.fetchall()

    # 🟩 Classify balls into line and length
    zone_maps = {
        "Full Toss": (-0.0909, 0.03636),
        "Yorker": (0.03636, 0.1636),
        "Full": (0.1636, 0.31818),
        "Good": (0.31818, 0.545454),
        "Short": (0.545454, 1.0)
    }
    zones = {}
    for length_label in zone_maps:
        for line_label in ["Wide Outside Off", "Outside Off", "Straight", "Leg"]:
            zones[(length_label, line_label)] = {"balls": 0, "runs": 0, "outs": 0}

    for b in balls:
        py, px = b["pitch_y"], b["pitch_x"]
        if px > 0.55:
            line_label = "Leg"
        elif 0.44 < px <= 0.55:
            line_label = "Straight"
        elif 0.26 < px <= 0.44:
            line_label = "Outside Off"
        else:
            line_label = "Wide Outside Off"

        length_label = next((l for l, (start, end) in zone_maps.items() if start <= py < end), "Unknown")

        zone = zones[(length_label, line_label)]
        zone["balls"] += 1
        zone["runs"] += b["runs"] or 0
        if b["dismissal_type"]:
            zone["outs"] += 1

    # 🟩 Determine best zone by ranking
    zone_scores = []
    for (length, line), stats in zones.items():
        if stats["balls"] == 0:
            continue
        rpb = stats["runs"] / stats["balls"]
        rpb_safe = max(rpb, 0.1)
        dismissal_pct = (stats["outs"] / stats["balls"]) * 100
        score = (dismissal_pct / 100) + (1 / rpb_safe)
        zone_scores.append((score, length, line, round(rpb, 2), round(dismissal_pct, 1)))

    zone_scores.sort(reverse=True)
    if zone_scores:
        _, best_length, best_line, _, _ = zone_scores[0]
    else:
        best_length, best_line = "Good", "Outside Off"

    recommended_zones = {"length": best_length, "line": best_line}
    summary = f"Use {recommended_type} bowlers, target {best_length} length and {best_line} line."

    # 🟩 Return detailed zone data
    zone_data = []
    for (length, line), stats in zones.items():
        if stats["balls"] == 0:
            continue
        rpb = stats["runs"] / stats["balls"]
        rpb_safe = max(rpb, 0.1)
        dismissal_pct = (stats["outs"] / stats["balls"]) * 100
        zone_data.append({
            "length": length,
            "line": line,
            "balls": stats["balls"],
            "runs": stats["runs"],
            "dismissals": stats["outs"],
            "avg_rpb": round(rpb, 2),
            "dismissal_pct": round(dismissal_pct, 1),
            "dot_pct": round((stats["balls"] - stats["runs"]) * 100 / stats["balls"], 1)
        })
    
    # 🟩 Determine best/worst zone for coloring
    zone_scores.sort(reverse=True)
    best_score = zone_scores[0][0] if zone_scores else None
    worst_score = zone_scores[-1][0] if zone_scores else None

    conn.close()
    return {
        "batter": batter_name,
        "avg_rpb_pace": detailed_stats["Pace"]["rpb"],
        "avg_rpb_medium": detailed_stats["Medium"]["rpb"],
        "avg_rpb_off_spin": detailed_stats["Off Spin"]["rpb"],
        "avg_rpb_leg_spin": detailed_stats["Leg Spin"]["rpb"],
        "dismissal_pct_pace": detailed_stats["Pace"]["dismissal_pct"],
        "dismissal_pct_medium": detailed_stats["Medium"]["dismissal_pct"],
        "dismissal_pct_off_spin": detailed_stats["Off Spin"]["dismissal_pct"],
        "dismissal_pct_leg_spin": detailed_stats["Leg Spin"]["dismissal_pct"],
        "dot_pct_pace": detailed_stats["Pace"]["dot_pct"],
        "dot_pct_medium": detailed_stats["Medium"]["dot_pct"],
        "dot_pct_off_spin": detailed_stats["Off Spin"]["dot_pct"],
        "dot_pct_leg_spin": detailed_stats["Leg Spin"]["dot_pct"],
        "recommended_bowler_type": recommended_type,
        "recommended_zones": recommended_zones,
        "summary": summary,
        "zone_data": zone_data,
        "best_zone_score": best_score,
        "worst_zone_score": worst_score
    }

@app.post("/generate-game-plan-pdf")
def generate_game_plan_pdf(payload: GamePlanPayload):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    bold = ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10)
    indent = ParagraphStyle(name='Indent', parent=styles['Normal'], leftIndent=20, fontSize=10)
    normal = styles['Normal']

    elements = []

    elements.append(Paragraph("<b>Game Plan Sheet</b>", styles['Title']))
    elements.append(Spacer(1, 10))

    # 🟩 Add Opponent Country up top
    opponent_country = payload.opponent_country if hasattr(payload, "opponent_country") else "Unknown"
    elements.append(Paragraph(f"<b>Opponent:</b> {opponent_country}", styles['Title']))
    elements.append(Spacer(1, 15))

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    for pid in payload.player_ids:
        cursor.execute("SELECT player_name FROM players WHERE player_id = ?", (pid,))
        player = cursor.fetchone()
        if not player:
            continue
        batter_name = player["player_name"]

        effectiveness = {}
        detailed_stats = {}
        has_data = False

        for style in ["Pace", "Medium", "Off Spin", "Leg Spin"]:
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END) AS legal_balls,
                    SUM(be.runs) AS runs,
                    SUM(CASE WHEN be.runs=0 AND be.wides=0 THEN 1 ELSE 0 END) AS dots,
                    SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS outs
                FROM ball_events be
                JOIN players bowl ON be.bowler_id = bowl.player_id
                WHERE be.batter_id = ? AND LOWER(bowl.bowling_style) = LOWER(?)
            """, (pid, style))
            row = cursor.fetchone()

            if not row or not row["legal_balls"]:
                detailed_stats[style] = {"balls": 0, "rpb": 0, "dot_pct": 0, "dismissal_pct": 0}
                effectiveness[style] = 0
                continue

            has_data = True  # ✅ Mark that we have at least some data
            balls = row["legal_balls"]
            runs = row["runs"] or 0
            outs = row["outs"] or 0
            rpb = round(runs / balls, 2)
            rpb_safe = max(rpb, 0.1)
            dot_pct = round((row["dots"] or 0) * 100 / balls, 1)
            out_pct = round(outs * 100 / balls, 1)

            effectiveness_score = (out_pct / 100) + (1 / rpb_safe)
            effectiveness[style] = effectiveness_score

            detailed_stats[style] = {
                "balls": balls,
                "rpb": rpb,
                "dot_pct": dot_pct,
                "dismissal_pct": out_pct
            }

        if not has_data:
            # 🟩 No data at all — simple line
            line = f"<b>{batter_name}</b>: No Data Available"
            elements.append(Paragraph(line, normal))
            elements.append(Spacer(1, 4))
            continue

        recommended_type = max(effectiveness, key=effectiveness.get)

        # 🟩 Brasil bowler selection (from frontend-provided bowler_ids)
        bowler_ids = payload.bowler_ids
        if bowler_ids:
            cursor.execute("""
                SELECT player_name, bowling_arm
                FROM players
                WHERE player_id IN ({})
                  AND LOWER(bowling_style) = LOWER(?)
            """.format(",".join(["?"] * len(bowler_ids))),
            bowler_ids + [recommended_type])
            bowlers = cursor.fetchall()
        else:
            bowlers = []

        bowler_names = ", ".join([f"{b['player_name']} ({b['bowling_arm']})" for b in bowlers]) or "No data"

        # 🟩 Zone analysis
        cursor.execute("""
            SELECT be.pitch_x, be.pitch_y, be.runs, be.dismissal_type
            FROM ball_events be
            JOIN players bowl ON be.bowler_id = bowl.player_id
            WHERE be.batter_id = ? AND LOWER(bowl.bowling_style) = LOWER(?)
              AND be.wides = 0 AND be.pitch_x IS NOT NULL AND be.pitch_y IS NOT NULL
        """, (pid, recommended_type))
        balls = cursor.fetchall()

        zone_maps = {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short of a": (0.545454, 1.0)
        }
        zones = {}
        for length_label in zone_maps:
            for line_label in ["Wide Outside Off", "Outside Off", "Off", "Middle/Leg"]:
                zones[(length_label, line_label)] = {"balls": 0, "runs": 0, "outs": 0}

        for b in balls:
            py, px = b["pitch_y"], b["pitch_x"]
            if px > 0.55:
                line_label = "Middle/Leg"
            elif 0.44 < px <= 0.55:
                line_label = "Off"
            elif 0.26 < px <= 0.44:
                line_label = "Outside Off"
            else:
                line_label = "Wide Outside Off"
            length_label = next((l for l, (start, end) in zone_maps.items() if start <= py < end), "Unknown")

            zone = zones[(length_label, line_label)]
            zone["balls"] += 1
            zone["runs"] += b["runs"] or 0
            if b["dismissal_type"]:
                zone["outs"] += 1

        zone_scores = []
        for (length, line), stats in zones.items():
            if stats["balls"] == 0:
                continue
            rpb = stats["runs"] / stats["balls"]
            rpb_safe = max(rpb, 0.1)
            dismissal_pct = (stats["outs"] / stats["balls"]) * 100
            score = (dismissal_pct / 100) + (1 / rpb_safe)
            zone_scores.append((score, length, line))

        zone_scores.sort(reverse=True)
        if zone_scores:
            _, best_length, best_line = zone_scores[0]
        else:
            best_length, best_line = "Good", "Outside Off"

        # 🟩 Output lines with formatting
        summary_line = f"<b>{batter_name}</b>: Use {recommended_type} bowlers, target {best_length} length and {best_line} line."
        #bowlers_line = f"Recommended bowlers: {bowler_names}"

        elements.append(Paragraph(summary_line, normal))
        #elements.append(Paragraph(bowlers_line, indent))
        elements.append(Spacer(1, 15))  # larger gap

    conn.close()
    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "inline; filename=game_plan_sheet.pdf"
    })

@app.get("/scorecard-player-detail")
def scorecard_player_detail(matchId: int, playerId: int):
    import sqlite3

    conn = sqlite3.connect("cricket_analysis.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 🧭 Shot locations for wagon wheel WITH dismissal info
    cursor.execute("""
        SELECT
            be.shot_x,
            be.shot_y,
            be.runs,
            CASE
                WHEN be.dismissed_player_id = be.batter_id
                     AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'retired out')
                THEN be.dismissal_type
                ELSE NULL
            END AS dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ?
          AND be.shot_x IS NOT NULL AND be.shot_y IS NOT NULL
    """, (matchId, playerId))
    shots = [dict(row) for row in cursor.fetchall()]

    # 🎯 Detailed breakdown: run counts, dot %, scoring shot %, avg intent
    cursor.execute("""
        SELECT
            COUNT(*) FILTER (WHERE be.wides = 0) AS balls_faced,
            SUM(be.runs) AS total_runs,
            SUM(CASE WHEN be.runs = 0 AND be.wides = 0 THEN 1 ELSE 0 END) AS dots,
            SUM(CASE WHEN be.runs = 1 THEN 1 ELSE 0 END) AS ones,
            SUM(CASE WHEN be.runs = 2 THEN 1 ELSE 0 END) AS twos,
            SUM(CASE WHEN be.runs = 3 THEN 1 ELSE 0 END) AS threes,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 5 THEN 1 ELSE 0 END) AS fives,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            ROUND(AVG(be.batting_intent_score), 2) AS avg_intent
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ?
    """, (matchId, playerId))

    row = cursor.fetchone()
    balls = row["balls_faced"] or 0
    dots = row["dots"] or 0
    scoring_shots = balls - dots
    scoring_pct = round((scoring_shots / balls) * 100, 1) if balls else 0.0

    breakdown = {
        "0": dots,
        "1": row["ones"] or 0,
        "2": row["twos"] or 0,
        "3": row["threes"] or 0,
        "4": row["fours"] or 0,
        "5": row["fives"] or 0,
        "6": row["sixes"] or 0,
    }

    conn.close()

    return {
        "shots": shots,
        "run_breakdown": breakdown,
        "scoring_pct": scoring_pct,
        "avg_intent": row["avg_intent"] or 0.0
    }

@app.get("/scorecard-bowler-detail")
def scorecard_bowler_detail(matchId: int, playerId: int):
    conn = sqlite3.connect("cricket_analysis.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 🟡 Get pitch map data (valid pitch points only)
    cursor.execute("""
        SELECT 
            be.pitch_x, 
            be.pitch_y, 
            be.runs, 
            be.wides,
            be.no_balls,
            CASE 
                WHEN LOWER(be.dismissal_type) IN ('bowled', 'caught', 'lbw', 'stumped', 'hit wicket') 
                THEN be.dismissal_type
                ELSE NULL
            END AS dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE be.bowler_id = ? 
        AND i.match_id = ?
        AND be.pitch_x IS NOT NULL 
        AND be.pitch_y IS NOT NULL
    """, (playerId, matchId))

    pitch_map = [dict(row) for row in cursor.fetchall()]

    # 🔢 Summary metrics (exclude run outs and similar)
    cursor.execute("""
        SELECT
            SUM(be.runs + be.wides + be.no_balls) AS runs,
            SUM(be.expected_runs + be.wides + be.no_balls) AS expected_runs,
            COUNT(*) FILTER (
                WHERE be.dismissal_type IS NOT NULL
                AND LOWER(be.dismissal_type) NOT IN ('run out', 'obstructing the field', 'retired', 'retired out', 'timed out', 'handled the ball')
                AND LOWER(be.dismissal_type) != 'not out'
            ) AS wickets,
            COUNT(*) FILTER (WHERE be.expected_wicket > 0) AS chance_events,
            SUM(be.expected_wicket) AS expected_wickets,
            COUNT(*) FILTER (WHERE be.wides = 0 AND be.no_balls = 0) AS balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE be.bowler_id = ? AND i.match_id = ?
    """, (playerId, matchId))

    row = cursor.fetchone()

    runs = row["runs"] or 0
    expected_runs = row["expected_runs"] or 0
    wickets = row["wickets"] or 0
    expected_wickets = row["expected_wickets"] or 0
    balls = row["balls"] or 0
    chances_from_expected = row["chance_events"] or 0

    # ✅ Final chance = expected wicket events + real dismissals
    chances_made = chances_from_expected + wickets
    real_wickets = expected_wickets + wickets

    real_econ = (expected_runs / (balls / 6)) if balls else 0
    real_sr = (balls / real_wickets) if real_wickets else None

    return {
        "pitch_map": pitch_map,
        "summary": {
            "runs_conceded": runs,
            "real_runs_conceded": round(expected_runs, 2),
            "chances_made": round(chances_made, 2),
            "wickets": wickets,
            "real_wickets": real_wickets,
            "real_economy": round(real_econ, 2) if balls else "–",
            "real_strike_rate": round(real_sr, 2) if real_sr else "–"
        }
    }

@app.post("/tournament-leaders/batting")
def get_batting_leaderboards(payload: dict):

    team_category = payload.get("team_category")
    tournament = payload.get("tournament")
    countries = payload.get("countries", [])

    if not team_category or not tournament or not countries:
        raise HTTPException(status_code=400, detail="Missing team_category, tournament, or countries.")

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Resolve tournament ID
    cursor.execute("SELECT tournament_id FROM tournaments WHERE tournament_name = ?", (tournament,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Tournament not found.")
    tournament_id = row["tournament_id"]

    # Resolve country IDs
    placeholders = ','.join('?' for _ in countries)
    cursor.execute(f"SELECT country_id, country_name FROM countries WHERE country_name IN ({placeholders})", countries)
    country_rows = cursor.fetchall()
    country_id_map = {r["country_id"]: r["country_name"] for r in country_rows}
    if not country_id_map:
        raise HTTPException(status_code=404, detail="Countries not found.")

    country_ids = list(country_id_map.keys())

    leaderboards = {}

    # Convert country_ids to country_names
    if country_ids:
        placeholders = ",".join("?" * len(country_ids))
        cursor.execute(f"""
            SELECT country_name FROM countries
            WHERE country_id IN ({placeholders})
        """, country_ids)
        country_names = [row["country_name"] for row in cursor.fetchall()]
    else:
        country_names = []

    if not country_names:
        return {"leaderboards": {}}


    # Most Runs
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            COUNT(DISTINCT i.match_id) AS matches,
            COUNT(DISTINCT i.innings_id) AS innings,
            SUM(be.runs) AS runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.batter_id = p.player_id
        WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?' * len(country_names))})
        GROUP BY be.batter_id
        ORDER BY runs DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most Runs"] = [dict(row) for row in cursor.fetchall()]



    # High Scores (allow repeated players, include not out status)
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            SUM(be.runs) AS runs,
            CASE 
                WHEN bd.dismissed_player_id IS NULL AND nbd.dismissal_type IS NULL
                THEN 1 ELSE 0
            END AS not_out
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.batter_id = p.player_id
        LEFT JOIN (
            SELECT innings_id, dismissed_player_id
            FROM ball_events
            WHERE dismissed_player_id IS NOT NULL
        ) bd ON bd.dismissed_player_id = be.batter_id AND bd.innings_id = i.innings_id
        LEFT JOIN (
            SELECT innings_id, player_id, dismissal_type
            FROM non_ball_dismissals
            WHERE LOWER(dismissal_type) != 'retired not out'
        ) nbd ON nbd.player_id = be.batter_id AND nbd.innings_id = i.innings_id
        WHERE m.tournament_id = ? AND i.batting_team IN ({','.join(['?'] * len(country_names))})
        GROUP BY be.batter_id, i.match_id
        ORDER BY runs DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["High Scores"] = [
        {
            "name": row["name"] + ("*" if row["not_out"] else ""),
            "high_score": row["runs"]
        }
        for row in cursor.fetchall()
    ]


    # Highest Averages
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            SUM(be.runs) AS total_runs,
            COUNT(DISTINCT CASE 
                WHEN be.dismissed_player_id = p.player_id THEN i.innings_id
                WHEN nbd.player_id IS NOT NULL THEN nbd.innings_id
            END) AS dismissals,
            ROUND(
                1.0 * SUM(be.runs) / 
                NULLIF(COUNT(DISTINCT CASE 
                    WHEN be.dismissed_player_id = p.player_id THEN i.innings_id
                    WHEN nbd.player_id IS NOT NULL THEN nbd.innings_id
                END), 0), 
                2
            ) AS average
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players p ON be.batter_id = p.player_id
        JOIN matches m ON i.match_id = m.match_id
        LEFT JOIN (
            SELECT innings_id, player_id
            FROM non_ball_dismissals
            WHERE LOWER(dismissal_type) != 'retired not out'
        ) AS nbd ON nbd.innings_id = i.innings_id AND nbd.player_id = p.player_id
        WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?' * len(country_names))})
        GROUP BY be.batter_id
        HAVING dismissals > 0
        ORDER BY average DESC
        LIMIT 10
    """, [tournament_id] + country_names)
    leaderboards["Highest Averages"] = [dict(row) for row in cursor.fetchall()]



    # Highest Strike Rates (min 30 balls faced, excluding wides)
    cursor.execute(f"""
        SELECT p.player_name AS name,
            SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END) AS balls_faced,
            ROUND(SUM(be.runs)*100.0 / NULLIF(SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END), 0), 2) AS strike_rate
        FROM ball_events be
        JOIN players p ON be.batter_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?'*len(country_names))})
        GROUP BY be.batter_id
        HAVING balls_faced >= 30
        ORDER BY strike_rate DESC LIMIT 10
    """, [tournament_id] + country_names)
    leaderboards["Highest Strike Rates"] = [dict(row) for row in cursor.fetchall()]

    # Most Fifties and Over
    cursor.execute(f"""
        SELECT p.player_name AS name, COUNT(*) AS fifties
        FROM (
            SELECT 
                be.batter_id,
                i.match_id,
                SUM(be.runs) AS runs
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?' * len(country_names))})
            GROUP BY be.batter_id, i.match_id
            HAVING SUM(be.runs) >= 50
        ) AS sub
        JOIN players p ON sub.batter_id = p.player_id
        GROUP BY sub.batter_id
        ORDER BY fifties DESC
        LIMIT 10
    """, [tournament_id] + country_names)
    leaderboards["Most Fifties and Over"] = [dict(row) for row in cursor.fetchall()]


    # Most Ducks
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            COUNT(*) AS ducks
        FROM (
            SELECT 
                be.batter_id,
                i.innings_id
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            LEFT JOIN non_ball_dismissals nbd 
                ON nbd.innings_id = i.innings_id AND nbd.player_id = be.batter_id
            WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?' * len(country_names))})
            GROUP BY be.batter_id, i.innings_id
            HAVING 
                SUM(CASE WHEN be.batter_id = be.batter_id THEN be.runs ELSE 0 END) = 0
                AND (
                    MAX(CASE 
                        WHEN be.dismissed_player_id = be.batter_id 
                            AND be.dismissal_type IS NOT NULL 
                            AND LOWER(be.dismissal_type) != 'not out'
                        THEN 1 ELSE 0 END) = 1
                    OR (
                        MAX(CASE 
                            WHEN nbd.dismissal_type IS NOT NULL 
                                AND LOWER(nbd.dismissal_type) NOT IN ('retired not out') 
                            THEN 1 ELSE 0 END) = 1
                    )
                )
        ) AS sub
        JOIN players p ON sub.batter_id = p.player_id
        GROUP BY sub.batter_id
        ORDER BY ducks DESC LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most Ducks"] = [dict(row) for row in cursor.fetchall()]


    # Most Fours
    cursor.execute(f"""
        SELECT p.player_name AS name, COUNT(*) AS fours
        FROM ball_events be
        JOIN players p ON be.batter_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.runs = 4 AND m.tournament_id = ? AND i.batting_team IN ({','.join('?'*len(country_names))})
        GROUP BY be.batter_id
        ORDER BY fours DESC LIMIT 10
    """, [tournament_id] + country_names)
    leaderboards["Most Fours"] = [dict(row) for row in cursor.fetchall()]

    # Most Sixes
    cursor.execute(f"""
        SELECT p.player_name AS name, COUNT(*) AS sixes
        FROM ball_events be
        JOIN players p ON be.batter_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.runs = 6 AND m.tournament_id = ? AND i.batting_team IN ({','.join('?'*len(country_names))})
        GROUP BY be.batter_id
        ORDER BY sixes DESC LIMIT 10
    """, [tournament_id] + country_names)
    leaderboards["Most Sixes"] = [dict(row) for row in cursor.fetchall()]

    # Highest Average Intent (Min 30 Balls Faced)
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            ROUND(AVG(be.batting_intent_score), 2) AS average_intent,
            COUNT(CASE WHEN be.wides = 0 THEN 1 END) AS balls_faced
        FROM ball_events be
        JOIN players p ON be.batter_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            be.batting_intent_score IS NOT NULL 
            AND m.tournament_id = ?
            AND i.batting_team IN ({','.join('?' * len(country_names))})
        GROUP BY be.batter_id
        HAVING balls_faced >= 30
        ORDER BY average_intent DESC LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Highest Average Intent"] = [dict(row) for row in cursor.fetchall()]


    # Scoring Shot % (Min 30 Balls Faced, excluding wides)
    cursor.execute(f"""
        SELECT 
            p.player_name AS name,
            ROUND(
                100.0 * SUM(CASE WHEN be.runs > 0 AND be.wides = 0 THEN 1 ELSE 0 END) /
                NULLIF(SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END), 0), 2
            ) AS scoring_shot_percentage,
            SUM(CASE WHEN be.wides = 0 THEN 1 ELSE 0 END) AS balls_faced
        FROM ball_events be
        JOIN players p ON be.batter_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.tournament_id = ? AND i.batting_team IN ({','.join('?' * len(country_names))})
        GROUP BY be.batter_id
        HAVING balls_faced >= 30
        ORDER BY scoring_shot_percentage DESC LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Highest Scoring Shot %"] = [dict(row) for row in cursor.fetchall()]




    conn.close()
    return leaderboards

@app.get("/venue-options")
def get_venue_options(tournament: str = None):
    import sqlite3, os

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT venue
        FROM matches
        WHERE 1=1
    """
    params = []

    if tournament:
        query += " AND tournament_id = (SELECT tournament_id FROM tournaments WHERE tournament_name = ?)"
        params.append(tournament)

    cursor.execute(query, params)
    venues = [row["venue"] for row in cursor.fetchall() if row["venue"]]

    grounds = set()
    times = set()

    for v in venues:
        if ',' in v:
            ground, time = [part.strip() for part in v.split(',', 1)]
            grounds.add(ground)
            times.add(time)
        else:
            grounds.add(v.strip())

    return {
        "grounds": sorted(grounds),
        "times": sorted(times)
    }

@app.post("/tournament-stats")
async def tournament_stats(request: Request):
    payload = await request.json()
    tournament = payload.get("tournament")
    team_category = payload.get("team_category")
    countries = payload.get("country", [])         # list[str]
    venues = payload.get("venue", [])              # list[str]
    times = payload.get("time_of_day", [])         # list[str]

    # 🧠 Combine venue and time into actual `m.venue` values
    venue_params = []
    if venues and times:
        for v in venues:
            for t in times:
                venue_params.append(f"{v}, {t}")
    elif venues:
        venue_params = [v.strip() for v in venues]
    elif times:
        # rare fallback: we want all venues with these times (partial match)
        pass  # optional: not supported unless needed

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT 
            m.venue,
            AVG(CASE WHEN i.innings = 1 THEN i.total_runs END) AS avg_score,
            SUM(CASE 
                WHEN i.innings = 1 AND i.batting_team = c.country_name AND m.winner_id = c.country_id 
                THEN 1 ELSE 0 END) AS bat1_wins,
            SUM(CASE 
                WHEN i.innings = 2 AND i.batting_team = c.country_name AND m.winner_id = c.country_id 
                THEN 1 ELSE 0 END) AS bat2_wins,
            COUNT(DISTINCT m.match_id) AS total_matches
        FROM matches m
        JOIN innings i ON m.match_id = i.match_id
        JOIN countries c ON i.batting_team = c.country_name
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE t.tournament_name = ?
    """
    params = [tournament]


    if countries:
        placeholders = ",".join(["?"] * len(countries))
        query += f" AND c.country_name IN ({placeholders})"
        params.extend(countries)

    if venue_params:
        placeholders = ",".join(["?"] * len(venue_params))
        query += f" AND m.venue IN ({placeholders})"
        params.extend(venue_params)

    query += " GROUP BY m.venue ORDER BY m.venue"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        result.append({
            "venue": row["venue"],
            "avg_score": round(row["avg_score"], 2) if row["avg_score"] is not None else None,
            "bat1_wins": row["bat1_wins"],
            "bat2_wins": row["bat2_wins"],
            "total_matches": row["total_matches"]
        })

    return JSONResponse(result)

@app.post("/tournament-leaders/bowling")
def get_tournament_bowling_leaders(payload: TournamentBowlingLeadersPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Resolve tournament_id
    cursor.execute("SELECT tournament_id FROM tournaments WHERE tournament_name = ?", (payload.tournament,))
    tournament_row = cursor.fetchone()
    if not tournament_row:
        return {}
    tournament_id = tournament_row["tournament_id"]

    # Resolve country names
    placeholders = ','.join(['?'] * len(payload.countries))
    country_names = payload.countries

    leaderboards = {}

    # Most Wickets
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            COUNT(*) AS wickets
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND be.dismissed_player_id IS NOT NULL
            AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
        GROUP BY be.bowler_id
        ORDER BY wickets DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most Wickets"] = [
        {
            "name": row["name"],
            "wickets": row["wickets"]
        }
        for row in cursor.fetchall()
    ]

    # Best Bowling Figures
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            i.match_id,
            i.batting_team AS opponent,
            SUM(be.runs + be.wides + be.no_balls) AS runs_conceded,
            COUNT(CASE 
                WHEN be.dismissed_player_id IS NOT NULL 
                    AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
                THEN 1 END) AS wickets
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        GROUP BY be.bowler_id, i.match_id
        HAVING wickets > 0
        ORDER BY wickets DESC, runs_conceded ASC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Best Bowling Figures"] = [
        {
            "name": row["name"],
            "figures": f"{row['wickets']}/{row['runs_conceded']}",
            "opponent": row["opponent"]
        }
        for row in cursor.fetchall()
    ]


    # Best Averages (min 4 wickets)
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            SUM(be.runs + be.wides + be.no_balls) * 1.0 /
                COUNT(CASE 
                    WHEN be.dismissed_player_id IS NOT NULL 
                        AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
                    THEN 1 END) AS avg_bowling,
            SUM(be.runs + be.wides + be.no_balls) AS total_runs,
            COUNT(CASE 
                WHEN be.dismissed_player_id IS NOT NULL 
                    AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
                THEN 1 END) AS total_wickets
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        GROUP BY be.bowler_id
        HAVING total_wickets >= 4
        ORDER BY avg_bowling ASC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Best Averages"] = [
        {
            "name": row["name"],
            "average": round(row["avg_bowling"], 2),
            "wickets": row["total_wickets"],
            "runs": row["total_runs"]
        }
        for row in cursor.fetchall()
    ]


    # Best Economy Rates
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            ROUND(SUM(be.runs + be.wides + be.no_balls) * 1.0 / (COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) / 6.0), 2) AS economy,
            SUM(be.runs + be.wides + be.no_balls) AS total_runs,
            COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) AS legal_deliveries
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        GROUP BY be.bowler_id
        HAVING legal_deliveries >= 30
        ORDER BY economy ASC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Best Economy Rates"] = [
        {
            "name": row["name"],
            "economy": row["economy"],
            "runs": row["total_runs"],
            "balls": row["legal_deliveries"]
        }
        for row in cursor.fetchall()
    ]

    # Best Strike Rates
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            ROUND(COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) * 1.0 /
                COUNT(CASE 
                    WHEN be.dismissed_player_id IS NOT NULL 
                        AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
                    THEN 1 END), 2) AS strike_rate,
            COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) AS legal_deliveries,
            COUNT(CASE 
                WHEN be.dismissed_player_id IS NOT NULL 
                    AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
                THEN 1 END) AS total_wickets
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        GROUP BY be.bowler_id
        HAVING total_wickets >= 4
        ORDER BY strike_rate ASC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Best Strike Rates"] = [
        {
            "name": row["name"],
            "strike_rate": row["strike_rate"],
            "balls": row["legal_deliveries"],
            "wickets": row["total_wickets"]
        }
        for row in cursor.fetchall()
    ]

    # 3+ Wicket Hauls
    cursor.execute(f"""
        SELECT 
            sub.bowler_id,
            p.player_name AS name,
            COUNT(*) AS three_wicket_hauls
        FROM (
            SELECT 
                be.bowler_id AS bowler_id,
                i.match_id AS match_id,
                COUNT(*) AS wickets
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE 
                m.tournament_id = ?
                AND i.bowling_team IN ({placeholders})
                AND be.dismissed_player_id IS NOT NULL
                AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'run out')
            GROUP BY be.bowler_id, i.match_id
            HAVING COUNT(*) >= 3
        ) AS sub
        JOIN players p ON sub.bowler_id = p.player_id
        GROUP BY sub.bowler_id
        ORDER BY three_wicket_hauls DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["3+ Wicket Hauls"] = [
        {
            "name": row["name"],
            "hauls": row["three_wicket_hauls"]
        }
        for row in cursor.fetchall()
    ]

    # Most Dot Balls
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            COUNT(*) AS dot_balls
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND be.runs = 0
            AND be.wides = 0
            AND be.no_balls = 0
        GROUP BY be.bowler_id
        ORDER BY dot_balls DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most Dot Balls"] = [
        {
            "name": row["name"],
            "dots": row["dot_balls"]
        }
        for row in cursor.fetchall()
    ]

    # Most Wides
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            SUM(be.wides) AS wides
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND be.wides > 0
        GROUP BY be.bowler_id
        ORDER BY wides DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most Wides"] = [
        {
            "name": row["name"],
            "wides": row["wides"]
        }
        for row in cursor.fetchall()
    ]

    # Most No Balls
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            SUM(be.no_balls) AS no_balls
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND be.no_balls > 0
        GROUP BY be.bowler_id
        ORDER BY no_balls DESC
        LIMIT 10
    """, [tournament_id] + country_names)

    leaderboards["Most No Balls"] = [
        {
            "name": row["name"],
            "no_balls": row["no_balls"]
        }
        for row in cursor.fetchall()
    ]

    # False Shot %
    cursor.execute(f"""
        SELECT 
            be.bowler_id,
            p.player_name AS name,
            COUNT(
                CASE 
                    WHEN (be.wides IS NULL OR be.wides = 0 AND be.no_balls IS NULL OR be.no_balls = 0)
                    THEN 1 END
            ) AS legal_deliveries,
            COUNT(
                CASE 
                    WHEN (be.wides IS NULL OR be.wides = 0) AND (
                        (be.dismissed_player_id IS NOT NULL 
                        AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'retired out', 'run out'))
                        OR be.edged = 1
                        OR (be.ball_missed = 1 AND LOWER(be.shot_selection) != 'leave')
                    )
                THEN 1 END
            ) AS false_shots,
            COUNT(*) AS total_balls,
            ROUND(
                COUNT(
                    CASE 
                        WHEN (be.wides IS NULL OR be.wides = 0) AND (
                            (be.dismissed_player_id IS NOT NULL 
                            AND LOWER(be.dismissal_type) NOT IN ('not out', 'retired hurt', 'retired out', 'run out'))
                            OR be.edged = 1
                            OR (be.ball_missed = 1 AND LOWER(be.shot_selection) != 'leave')
                        )
                    THEN 1 END
                ) * 100.0 / COUNT(*), 2
            ) AS false_shot_percent
        FROM ball_events be
        JOIN players p ON be.bowler_id = p.player_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE 
            m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        GROUP BY be.bowler_id
        HAVING legal_deliveries >= 30
        ORDER BY false_shot_percent DESC
        LIMIT 10;
    """, [tournament_id] + country_names)

    leaderboards["False Shot %"] = [
        {
            "name": row["name"],
            "false_shots": row["false_shots"],
            "deliveries": row["total_balls"],
            "false_shot_percent": row["false_shot_percent"]
        }
        for row in cursor.fetchall()
    ]






    return leaderboards

@app.post("/tournament-leaders/fielding")
def get_tournament_fielding_leaders(payload: TournamentFieldingLeadersPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Resolve tournament_id
    cursor.execute("SELECT tournament_id FROM tournaments WHERE tournament_name = ?", (payload.tournament,))
    tournament_row = cursor.fetchone()
    if not tournament_row:
        return {}
    tournament_id = tournament_row["tournament_id"]

    # Country filter
    country_names = payload.countries
    placeholders = ','.join(['?'] * len(country_names))

    leaderboards = {}

    # 1. Most Catches (excluding wicketkeepers)
    cursor.execute(f"""
        WITH non_wk_fielders AS (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be ON fc.ball_id = be.ball_id
            WHERE LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
        )
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS catches
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) = 'caught'
        AND be.fielder_id IS NOT NULL
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        AND be.fielder_id NOT IN (SELECT fielder_id FROM non_wk_fielders)
        GROUP BY be.fielder_id
        ORDER BY catches DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["Most Catches"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["catches"]
        }
        for row in cursor.fetchall()
    ]

    # 2. Most Run Outs (excluding wicketkeepers)
    cursor.execute(f"""
        WITH non_wk_fielders AS (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be ON fc.ball_id = be.ball_id
            WHERE LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
        )
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS run_outs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) = 'run out'
        AND be.fielder_id IS NOT NULL
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        AND be.fielder_id NOT IN (SELECT fielder_id FROM non_wk_fielders)
        GROUP BY be.fielder_id
        ORDER BY run_outs DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["Most Run Outs"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["run_outs"]
        }
        for row in cursor.fetchall()
    ]

    cursor.execute(f"""
        WITH non_wk_fielders AS (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be ON fc.ball_id = be.ball_id
            WHERE LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
        )
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS dismissals
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) IN ('caught', 'run out')
        AND be.fielder_id IS NOT NULL
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        AND be.fielder_id NOT IN (SELECT fielder_id FROM non_wk_fielders)
        GROUP BY be.fielder_id
        ORDER BY dismissals DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["Most Dismissals"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["dismissals"]
        }
        for row in cursor.fetchall()
    ]

        # 4. Best Conversion Rate (excluding wicketkeepers, % format)
    cursor.execute(f"""
        WITH non_keepers AS (
            SELECT p.player_id
            FROM players p
            WHERE p.player_id NOT IN (
                SELECT DISTINCT be.fielder_id
                FROM ball_events be
                WHERE LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
                AND be.fielder_id IS NOT NULL
            )
        ),
        non_wk_dismissals AS (
            SELECT be.fielder_id, COUNT(*) AS dismissals
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE LOWER(be.dismissal_type) IN ('caught', 'run out')
            AND m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND be.fielder_id IN (SELECT player_id FROM non_keepers)
            GROUP BY be.fielder_id
        ),
        non_wk_misses AS (
            SELECT fc.fielder_id, COUNT(*) AS misses
            FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON be.ball_id = bfe.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE bfe.event_id IN (6, 7, 8)
            AND m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            AND fc.fielder_id IN (SELECT player_id FROM non_keepers)
            GROUP BY fc.fielder_id
        )
        SELECT 
            p.player_id AS fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COALESCE(d.dismissals, 0) AS dismissals,
            COALESCE(m.misses, 0) AS misses,
            COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0) AS total_chances,
            ROUND(
                100.0 * COALESCE(d.dismissals, 0) /
                NULLIF(COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0), 0), 1
            ) AS conversion_rate
        FROM players p
        JOIN countries c ON p.country_id = c.country_id
        JOIN non_keepers nk ON p.player_id = nk.player_id
        LEFT JOIN non_wk_dismissals d ON p.player_id = d.fielder_id
        LEFT JOIN non_wk_misses m ON p.player_id = m.fielder_id
        WHERE (COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0)) > 0
        ORDER BY conversion_rate DESC
        LIMIT 10;
    """, [tournament_id] + country_names + [tournament_id] + country_names)



    leaderboards["Best Conversion Rate"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["conversion_rate"]  # % format (e.g. 87.5)
        }
        for row in cursor.fetchall()
    ]

    # 5. Cleanest Hands (excluding wicketkeepers)
    cursor.execute(f"""
        SELECT 
            fc.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            SUM(CASE WHEN bfe.event_id = 1 THEN 1 ELSE 0 END) AS clean_pickups,
            COUNT(*) AS total_fielding_events,
            ROUND(
                100.0 * SUM(CASE WHEN bfe.event_id = 1 THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) AS clean_hands_pct
        FROM ball_fielding_events bfe
        JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
        JOIN ball_events be ON be.ball_id = bfe.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON fc.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE i.bowling_team IN ({placeholders})
          AND m.tournament_id = ?
          AND fc.fielder_id NOT IN (
              SELECT DISTINCT fc2.fielder_id
              FROM fielding_contributions fc2
              JOIN ball_events be2 ON fc2.ball_id = be2.ball_id
              WHERE LOWER(be2.fielding_style) IN ('wk normal', 'wk dive')
          )
        GROUP BY fc.fielder_id
        HAVING total_fielding_events > 20
        ORDER BY clean_hands_pct DESC
        LIMIT 10
    """, country_names + [tournament_id])

    leaderboards["Cleanest Hands"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["clean_hands_pct"]  # % format
        }
        for row in cursor.fetchall()
    ]

        # 6. WK Catches (only 'catching' by known keepers)
    cursor.execute(f"""
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS wk_catches
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) = 'caught'
        AND be.fielder_id IN (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be2 ON fc.ball_id = be2.ball_id
            WHERE LOWER(be2.fielding_style) IN ('wk normal', 'wk dive')
        )
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        GROUP BY be.fielder_id
        ORDER BY wk_catches DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["WK Catches"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["wk_catches"]
        }
        for row in cursor.fetchall()
    ]

        # 7. WK Stumpings
    cursor.execute(f"""
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS wk_stumpings
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) = 'stumped'
        AND be.fielder_id IN (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be2 ON fc.ball_id = be2.ball_id
            WHERE LOWER(be2.fielding_style) IN ('wk normal', 'wk dive')
        )
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        GROUP BY be.fielder_id
        ORDER BY wk_stumpings DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["WK Stumpings"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["wk_stumpings"]
        }
        for row in cursor.fetchall()
    ]

        # 8. WK Dismissals
    cursor.execute(f"""
        SELECT 
            be.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COUNT(*) AS wk_dismissals
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON be.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.dismissal_type) IN ('caught', 'run out', 'stumped')
        AND be.fielder_id IN (
            SELECT DISTINCT fc.fielder_id
            FROM fielding_contributions fc
            JOIN ball_events be2 ON fc.ball_id = be2.ball_id
            WHERE LOWER(be2.fielding_style) IN ('wk normal', 'wk dive')
        )
        AND i.bowling_team IN ({placeholders})
        AND m.tournament_id = ?
        GROUP BY be.fielder_id
        ORDER BY wk_dismissals DESC
        LIMIT 10
    """, country_names + [tournament_id])


    leaderboards["WK Dismissals"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["wk_dismissals"]
        }
        for row in cursor.fetchall()
    ]

    # Wicket Keeper Conversion
    cursor.execute(f"""
        WITH keepers AS (
            SELECT DISTINCT be.fielder_id
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
            AND be.fielder_id IS NOT NULL
            AND m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
        ),
        wk_dismissals AS (
            SELECT be.fielder_id, COUNT(*) AS dismissals
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE LOWER(be.dismissal_type) IN ('caught', 'run out', 'stumped')
            AND be.fielder_id IN (SELECT fielder_id FROM keepers)
            AND m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            GROUP BY be.fielder_id
        ),
        wk_misses AS (
            SELECT fc.fielder_id, COUNT(*) AS misses
            FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON be.ball_id = bfe.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE bfe.event_id IN (6, 7, 8, 15)
            AND LOWER(be.fielding_style) IN ('wk normal', 'wk dive')
            AND fc.fielder_id IN (SELECT fielder_id FROM keepers)
            AND m.tournament_id = ?
            AND i.bowling_team IN ({placeholders})
            GROUP BY fc.fielder_id
        )
        SELECT 
            p.player_id AS fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            COALESCE(d.dismissals, 0) AS wk_dismissals,
            COALESCE(m.misses, 0) AS wk_misses,
            COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0) AS total_chances,
            ROUND(
                100.0 * COALESCE(d.dismissals, 0) /
                NULLIF(COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0), 0), 1
            ) AS wk_conversion_rate
        FROM players p
        JOIN countries c ON p.country_id = c.country_id
        JOIN keepers k ON p.player_id = k.fielder_id
        LEFT JOIN wk_dismissals d ON p.player_id = d.fielder_id
        LEFT JOIN wk_misses m ON p.player_id = m.fielder_id
        WHERE (COALESCE(d.dismissals, 0) + COALESCE(m.misses, 0)) > 0
        ORDER BY wk_conversion_rate DESC
        LIMIT 10
    """, [tournament_id] + country_names + [tournament_id] + country_names + [tournament_id] + country_names)


    leaderboards["Best WK Conversion Rate"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["wk_conversion_rate"]  # in percent format
        }
        for row in cursor.fetchall()
    ]

    # 10. WK Cleanest Hands
    cursor.execute(f"""
        SELECT 
            fc.fielder_id,
            p.player_name AS name,
            c.country_name AS country,
            SUM(CASE WHEN bfe.event_id = 1 THEN 1 ELSE 0 END) AS clean_pickups,
            COUNT(*) AS total_fielding_events,
            ROUND(
                100.0 * SUM(CASE WHEN bfe.event_id = 1 THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) AS wk_clean_hands_pct
        FROM ball_fielding_events bfe
        JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
        JOIN ball_events be ON be.ball_id = bfe.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        JOIN players p ON fc.fielder_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE LOWER(be.fielding_style) IN ('catching', 'wk normal', 'wk dive')
          AND fc.fielder_id IN (
              SELECT DISTINCT fc2.fielder_id
              FROM fielding_contributions fc2
              JOIN ball_events be2 ON fc2.ball_id = be2.ball_id
              WHERE LOWER(be2.fielding_style) IN ('wk normal', 'wk dive')
          )
          AND i.bowling_team IN ({placeholders})
          AND m.tournament_id = ?
        GROUP BY fc.fielder_id
        HAVING total_fielding_events > 30
        ORDER BY wk_clean_hands_pct DESC
        LIMIT 10;
    """, country_names + [tournament_id])

    leaderboards["WK Cleanest Hands"] = [
        {
            "name": row["name"],
            "country": row["country"],
            "value": row["wk_clean_hands_pct"]  # % format
        }
        for row in cursor.fetchall()
    ]


    return leaderboards

@app.post("/tournament-standings")
def get_tournament_standings(payload: dict):
    import sqlite3

    team_category = payload["team_category"]
    tournament = payload["tournament"]

    conn = sqlite3.connect("cricket_analysis.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Query with winner name and adjusted overs
    cur.execute("""
        SELECT 
            i.innings_id,
            i.batting_team,
            i.bowling_team,
            i.overs_bowled,
            i.wickets,
            i.total_runs,
            i.innings,
            m.match_id,
            m.result,
            m.winner_id,
            cw.country_name AS winner_name,
            m.adjusted_overs
        FROM innings i
        JOIN matches m ON i.match_id = m.match_id
        LEFT JOIN countries cw ON m.winner_id = cw.country_id
        WHERE m.tournament_id = (SELECT tournament_id FROM tournaments WHERE tournament_name = ?)
    """, (tournament,))

    innings_data = cur.fetchall()
    team_stats = {}

    for row in innings_data:
        team = row["batting_team"]
        opp = row["bowling_team"]
        match_id = row["match_id"]
        runs = row["total_runs"]
        wickets = row["wickets"]
        innings = row["innings"]
        overs_bowled = row["overs_bowled"]
        result = row["result"]
        winner_name = row["winner_name"]
        adjusted_overs = row["adjusted_overs"] or 20.0

        # NRR-safe overs faced logic
        is_chasing = innings == 2
        lost_while_chasing = is_chasing and winner_name and winner_name != team
        was_all_out = wickets >= 10

        # Determine correct overs faced for this innings
        if innings == 1 and overs_bowled > adjusted_overs:
            # Rain came after full first innings; use what was actually bowled
            overs_faced = overs_bowled
        elif was_all_out or lost_while_chasing:
            # Use adjusted overs for second innings loss or all out
            overs_faced = adjusted_overs
        else:
            overs_faced = overs_bowled

        # Init batting team
        if team not in team_stats:
            team_stats[team] = {
                "played": 0, "wins": 0, "no_results": 0, "points": 0,
                "runs_scored": 0, "overs_faced": 0.0,
                "runs_conceded": 0, "overs_bowled": 0.0
            }

        team_stats[team]["played"] += 1

        if result == "no result":
            team_stats[team]["no_results"] += 1
            team_stats[team]["points"] += 1
        elif winner_name == team:
            team_stats[team]["wins"] += 1
            team_stats[team]["points"] += 2
        # ❌ no manual losses += 1 here

        team_stats[team]["runs_scored"] += runs
        team_stats[team]["overs_faced"] += overs_faced

        # Init bowling team
        if opp not in team_stats:
            team_stats[opp] = {
                "played": 0, "wins": 0, "no_results": 0, "points": 0,
                "runs_scored": 0, "overs_faced": 0.0,
                "runs_conceded": 0, "overs_bowled": 0.0
            }

        team_stats[opp]["runs_conceded"] += runs
        team_stats[opp]["overs_bowled"] += overs_faced  # key: same overs as batting team faced

    # Format final table
    table = []
    for team, data in team_stats.items():
        if data["overs_faced"] == 0 or data["overs_bowled"] == 0:
            nrr = 0.0
        else:
            nrr = (data["runs_scored"] / data["overs_faced"]) - (data["runs_conceded"] / data["overs_bowled"])

        losses = data["played"] - data["wins"] - data["no_results"]

        table.append({
            "team": team,
            "played": data["played"],
            "wins": data["wins"],
            "losses": losses,
            "no_results": data["no_results"],
            "points": data["points"],
            "nrr": round(nrr, 3)
        })

    table.sort(key=lambda x: (-x["points"], -x["nrr"], x["team"].lower()))
    return table

@app.get("/venue-insights")
def venue_insights(
    ground: str = Query(..., description="Ground name (e.g., 'Wankhede Stadium')"),
    time_of_day: Optional[str] = Query(None, description="Optional time of day, e.g., 'Day' / 'Night' if you store it"),
    tournament: Optional[str] = None,
    team_category: Optional[str] = None
):
    import sqlite3, os

    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Build concrete venue values to match how you store them ("Ground, Time") or just "Ground"
    if time_of_day:
        venues_to_match = [f"{ground}, {time_of_day}"]
    else:
        # Match either exact "Ground" or any "Ground, <time>" variants
        # We'll use two predicates: exact OR startswith "Ground, "
        venues_to_match = None  # handled via SQL LIKE for the ", time" variant

    params = []
    where = ["1=1"]

    if venues_to_match:
        where.append("m.venue = ?")
        params.append(venues_to_match[0])
    else:
        where.append("(m.venue = ? OR m.venue LIKE ?)")
        params.extend([ground, f"{ground}, %"])

    if tournament:
        where.append("m.tournament_id = (SELECT tournament_id FROM tournaments WHERE tournament_name = ?)")
        params.append(tournament)

    # Optional team_category filter by team names (your data encodes category in country_name)
    # We include matches where either side's name matches the category, excluding "training" unless explicitly chosen.
    if team_category:
        where.append("(LOWER(ca.country_name) LIKE ? OR LOWER(cb.country_name) LIKE ?)")
        params.extend([f"%{team_category.lower()}%", f"%{team_category.lower()}%"])

    # ---------- Average 1st-innings total ----------
    c.execute(f"""
        SELECT AVG(i.total_runs) AS avg_first_innings
        FROM matches m
        JOIN innings i ON i.match_id = m.match_id AND i.innings = 1
        JOIN countries ca ON ca.country_id = m.team_a
        JOIN countries cb ON cb.country_id = m.team_b
        WHERE {' AND '.join(where)}
    """, params)
    row = c.fetchone()
    avg_first_innings = round(row["avg_first_innings"], 2) if row and row["avg_first_innings"] is not None else None

    # ---------- Win rate when batting first ----------
    # Map the batting team (innings=1) to a country_id and compare with winner_id
    c.execute(f"""
        SELECT 
            SUM(CASE WHEN cw.country_id = m.winner_id THEN 1 ELSE 0 END) AS wins_batting_first,
            COUNT(CASE WHEN m.winner_id IS NOT NULL THEN 1 END) AS decided_matches
        FROM matches m
        JOIN innings i1 ON i1.match_id = m.match_id AND i1.innings = 1
        JOIN countries cw ON cw.country_name = i1.batting_team
        JOIN countries ca ON ca.country_id = m.team_a
        JOIN countries cb ON cb.country_id = m.team_b
        WHERE {' AND '.join(where)}
    """, params)
    row = c.fetchone()
    wins_bat_first = row["wins_batting_first"] or 0
    decided = row["decided_matches"] or 0
    bat_first_win_rate = round((wins_bat_first / decided) * 100, 2) if decided > 0 else None

    # ---------- Toss decision distribution & most common ----------
    c.execute(f"""
        SELECT COALESCE(m.toss_decision, 'unknown') AS decision, COUNT(*) AS cnt
        FROM matches m
        JOIN countries ca ON ca.country_id = m.team_a
        JOIN countries cb ON cb.country_id = m.team_b
        WHERE {' AND '.join(where)}
        GROUP BY COALESCE(m.toss_decision, 'unknown')
        ORDER BY cnt DESC
    """, params)
    toss_rows = c.fetchall()
    toss_distribution = {r["decision"]: r["cnt"] for r in toss_rows}
    most_common_toss_decision = (toss_rows[0]["decision"] if toss_rows else None)

    conn.close()
    return JSONResponse({
        "ground": ground,
        "time_of_day": time_of_day,
        "avg_first_innings": avg_first_innings,
        "bat_first_win_rate_pct": bat_first_win_rate,         # e.g., 44.12
        "toss_distribution": toss_distribution,                # {"bat": 12, "field": 18, "unknown": 1}
        "most_common_toss_decision": most_common_toss_decision # "field" / "bat" / "unknown"
    })

@app.post("/opposition-key-players")
def opposition_key_players(payload: OppKeyPlayersPayload):
    import sqlite3, os
    from fastapi.responses import JSONResponse

    team_category = payload.team_category
    opponent_country = payload.opponent_country
    min_balls = payload.min_balls
    min_overs = payload.min_overs

    conn = _db()
    c = conn.cursor()

    # roster
    c.execute("""
        SELECT p.player_id, p.player_name, p.role, p.bowling_style
        FROM players p
        JOIN countries ctry ON p.country_id = ctry.country_id
        WHERE ctry.country_name = ?
    """, (opponent_country,))
    roster = c.fetchall()
    if not roster:
        conn.close()
        return JSONResponse({"batters": [], "bowlers": []})

    player_ids = [r["player_id"] for r in roster]
    placeholders = ",".join(["?"] * len(player_ids))

    # --- BATTERS: SR desc, then Avg desc, then runs
    c.execute(f"""
        WITH batter_raw AS (
            SELECT 
                be.batter_id AS pid,
                SUM(COALESCE(be.runs,0)) AS runs,
                COUNT(CASE WHEN be.wides = 0 THEN 1 END) AS balls_faced,
                SUM(
                CASE
                    WHEN be.dismissed_player_id = be.batter_id
                    AND LOWER(COALESCE(be.dismissal_type,'')) NOT IN (
                        'not out','run out','retired hurt','retired out','obstructing the field'
                    )
                    THEN 1 ELSE 0
                END
                ) AS dismissals
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE be.batter_id IN ({placeholders})
            GROUP BY be.batter_id                 -- ✅ key fix
        )
        SELECT 
            r.pid AS player_id,
            p.player_name,
            COALESCE(r.runs,0) AS runs,
            COALESCE(r.balls_faced,0) AS balls_faced,
            COALESCE(r.dismissals,0) AS dismissals,
            CASE WHEN COALESCE(r.balls_faced,0) > 0 THEN ROUND(r.runs * 100.0 / r.balls_faced, 2) ELSE 0 END AS strike_rate,
            CASE WHEN COALESCE(r.dismissals,0) > 0 THEN ROUND(r.runs * 1.0 / r.dismissals, 2) ELSE NULL END AS average
        FROM batter_raw r
        JOIN players p ON p.player_id = r.pid
        WHERE COALESCE(r.balls_faced,0) >= ?
        ORDER BY strike_rate DESC, COALESCE(average, -1) DESC, runs DESC
        LIMIT 3
    """, (*player_ids, min_balls))
    top_batters = [dict(row) for row in c.fetchall()]

    # --- BOWLERS: Wickets desc, tie-break Eco asc, then overs desc
    c.execute(f"""
        WITH bowler_raw AS (
            SELECT
                be.bowler_id AS pid,
                COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) AS legal_balls,
                SUM(COALESCE(be.runs,0) + COALESCE(be.wides,0) + COALESCE(be.no_balls,0)) AS runs_conceded,
                SUM(
                CASE
                    WHEN be.dismissed_player_id = be.batter_id
                    AND LOWER(COALESCE(be.dismissal_type,'')) NOT IN (
                        'not out','run out','retired hurt','retired out','obstructing the field'
                    )
                    THEN 1 ELSE 0
                END
                ) AS wickets
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE be.bowler_id IN ({placeholders})
            GROUP BY be.bowler_id                 -- ✅ key fix
        )
        SELECT
            r.pid AS player_id,
            p.player_name,
            COALESCE(r.legal_balls,0) AS legal_balls,
            COALESCE(r.runs_conceded,0) AS runs_conceded,
            COALESCE(r.wickets,0) AS wickets,
            CASE WHEN COALESCE(r.legal_balls,0) > 0 THEN ROUND(r.runs_conceded * 6.0 / r.legal_balls, 2) ELSE NULL END AS economy,
            ROUND(COALESCE(r.legal_balls,0) / 6.0, 1) AS overs
        FROM bowler_raw r
        JOIN players p ON p.player_id = r.pid
        WHERE COALESCE(r.legal_balls,0) >= (? * 6)
        ORDER BY wickets DESC, COALESCE(economy, 9999) ASC, overs DESC
        LIMIT 3
    """, (*player_ids, min_overs))
    top_bowlers = [dict(row) for row in c.fetchall()]

    conn.close()
    return JSONResponse({"batters": top_batters, "bowlers": top_bowlers})

# --- Helper: normalize bowling style buckets used in the UI ---
STYLE_CASE_SQL = """
CASE
  WHEN LOWER(bowl.bowling_style) = 'pace'     THEN 'Pace'
  WHEN LOWER(bowl.bowling_style) = 'medium'   THEN 'Medium'
  WHEN LOWER(bowl.bowling_style) = 'off spin' THEN 'Off Spin'
  WHEN LOWER(bowl.bowling_style) = 'leg spin' THEN 'Leg Spin'
  ELSE 'Unknown'
END
"""

def _phase_case(alias: str = "be") -> str:
    # Using your boolean phase flags on ball_events
    return f"""
    CASE
      WHEN {alias}.is_powerplay = 1    THEN 'Powerplay'
      WHEN {alias}.is_death_overs = 1  THEN 'Death'
      ELSE 'Middle'
    END
    """

def _empty_strengths_payload() -> Dict[str, Any]:
    return {
        "batting": {
            "strengths": [],
            "weaknesses": [],
            "by_style": [],
            "by_phase": [],
        },
        "bowling": {
            "strengths": [],
            "weaknesses": [],
            "by_style": [],
            "by_phase": [],
        },
    }

@app.post("/opposition-strengths")
def opposition_strengths(payload: OppositionStrengthsPayload = Body(...)) -> JSONResponse:
    """
    Opposition strengths/weaknesses for a country across all recorded T20 balls.
    - Batting by bowler type & phase (strike rate, dot%, boundary%, outs/ball)
    - Bowling by phase & type (econ, dot%, wickets/ball, boundary%)
    Uses Option A: filter AFTER aggregation via an outer WHERE.
    """
    opponent = (payload.opponent_country or "").strip()
    if not opponent:
        return JSONResponse(_empty_strengths_payload())

    conn = _db()
    c = conn.cursor()

    # ---- Get player id lists for the opponent (batter ids and bowler ids) ----
    # We don't constrain by role; actual participation is filtered by ball_events usage.
    c.execute("""
        SELECT p.player_id
        FROM players p
        JOIN countries ctry ON p.country_id = ctry.country_id
        WHERE ctry.country_name = ?
    """, (opponent,))
    opp_ids = [r[0] for r in c.fetchall()]

    if not opp_ids:
        conn.close()
        return JSONResponse(_empty_strengths_payload())

    in_placeholders = ",".join(["?"] * len(opp_ids))

    # =========================
    # BAT T I N G  (opponent batting vs all bowlers)
    # =========================

    # --- Batting by bowler style ---
    # Definitions (to mirror your app logic):
    #   balls        = COUNT where wides=0  (includes no-balls as a ball faced, matching your earlier logic)
    #   runs_total   = SUM(be.runs)         (only off-the-bat runs)
    #   dots         = wides=0 AND runs=0
    #   boundaries   = wides=0 AND runs>=4
    #   outs         = dismissal_type not null/empty (we treat run-outs as outs for the batter)
    # NOTE: No HAVING; we filter by "balls >= ?" in the OUTER WHERE per Option A.
    c.execute(f"""
        WITH raw AS (
          SELECT
            {STYLE_CASE_SQL} AS style_norm,
            be.wides,
            be.no_balls,
            be.runs,
            be.dismissal_type
          FROM ball_events be
          JOIN players bat ON be.batter_id = bat.player_id
          JOIN players bowl ON be.bowler_id = bowl.player_id
          WHERE be.batter_id IN ({in_placeholders})
        ),
        agg AS (
          SELECT
            style_norm,
            SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END)                                AS balls,
            SUM(runs)                                                                  AS runs_total,
            SUM(CASE WHEN wides = 0 AND runs = 0 THEN 1 ELSE 0 END)                    AS dots,
            SUM(CASE WHEN wides = 0 AND runs >= 4 THEN 1 ELSE 0 END)                   AS boundaries,
            SUM(CASE WHEN dismissal_type IS NOT NULL AND TRIM(dismissal_type) <> '' 
                     THEN 1 ELSE 0 END)                                                AS outs
          FROM raw
          GROUP BY style_norm
        )
        SELECT
          style_norm,
          balls,
          ROUND(runs_total * 100.0 / NULLIF(balls, 0), 1)                              AS strike_rate,
          ROUND(dots * 100.0 / NULLIF(balls, 0), 1)                                    AS dot_pct,
          ROUND(boundaries * 100.0 / NULLIF(balls, 0), 1)                              AS boundary_pct,
          ROUND(outs * 1.0 / NULLIF(balls, 0), 4)                                      AS outs_perc_ball
        FROM agg
        WHERE balls >= ?
        ORDER BY style_norm
    """, (*opp_ids, payload.min_balls_style))
    batting_by_style = [dict(r) for r in c.fetchall()]

    # --- Batting by phase ---
    phase_case = _phase_case("be")
    c.execute(f"""
        WITH raw AS (
          SELECT
            {_phase_case("be")} AS phase,
            be.wides,
            be.no_balls,
            be.runs
          FROM ball_events be
          WHERE be.batter_id IN ({in_placeholders})
        ),
        agg AS (
          SELECT
            phase,
            SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END)                       AS balls,
            SUM(runs)                                                        AS runs_total,
            SUM(CASE WHEN wides = 0 AND runs = 0 THEN 1 ELSE 0 END)          AS dots,
            SUM(CASE WHEN wides = 0 AND runs >= 4 THEN 1 ELSE 0 END)         AS boundaries
          FROM raw
          GROUP BY phase
        )
        SELECT
          phase,
          balls,
          ROUND(runs_total * 100.0 / NULLIF(balls, 0), 1)                    AS strike_rate,
          ROUND(dots * 100.0 / NULLIF(balls, 0), 1)                          AS dot_pct,
          ROUND(boundaries * 100.0 / NULLIF(balls, 0), 1)                    AS boundary_pct
        FROM agg
        WHERE balls >= ?
        ORDER BY CASE phase
                   WHEN 'Powerplay' THEN 1
                   WHEN 'Middle'    THEN 2
                   WHEN 'Death'     THEN 3
                   ELSE 4
                 END
    """, (*opp_ids, payload.min_balls_phase))
    batting_by_phase = [dict(r) for r in c.fetchall()]

    # --- Batting strengths/weaknesses (simple heuristics) ---
    # Strengths: highest strike rate rows by style/phase (top 2 each if present)
    # Weaknesses: highest dot% rows and/or lowest strike rate (top 2 each, merged unique).
    strengths_bat: List[str] = []
    weaknesses_bat: List[str] = []

    if batting_by_style:
        st_sorted = sorted(batting_by_style, key=lambda x: (x["strike_rate"] or 0), reverse=True)
        strengths_bat += [f"Scoring well vs {st_sorted[0]['style_norm']} (SR {st_sorted[0]['strike_rate']})"]
        if len(st_sorted) > 1:
            strengths_bat += [f"Also solid vs {st_sorted[1]['style_norm']} (SR {st_sorted[1]['strike_rate']})"]

        dt_sorted = sorted(batting_by_style, key=lambda x: (x["dot_pct"] or 0), reverse=True)
        weaknesses_bat += [f"High dot% vs {dt_sorted[0]['style_norm']} ({dt_sorted[0]['dot_pct']}%)"]
        lr_sorted = sorted(batting_by_style, key=lambda x: (x["strike_rate"] or 0))
        weaknesses_bat += [f"Lower SR vs {lr_sorted[0]['style_norm']} (SR {lr_sorted[0]['strike_rate']})"]

    if batting_by_phase:
        stp_sorted = sorted(batting_by_phase, key=lambda x: (x["strike_rate"] or 0), reverse=True)
        strengths_bat += [f"Best phase: {stp_sorted[0]['phase']} (SR {stp_sorted[0]['strike_rate']})"]
        dtp_sorted = sorted(batting_by_phase, key=lambda x: (x["dot_pct"] or 0), reverse=True)
        weaknesses_bat += [f"Most dots in {dtp_sorted[0]['phase']} ({dtp_sorted[0]['dot_pct']}%)"]

    # Deduplicate while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    strengths_bat = _dedup(strengths_bat)[:4]
    weaknesses_bat = _dedup(weaknesses_bat)[:4]

    # =========================
    # B O W L I N G  (opponent bowling vs all batters)
    # =========================

    # by phase
    c.execute(f"""
        WITH raw AS (
          SELECT
            {_phase_case("be")} AS phase,
            be.runs + be.wides + be.no_balls                                 AS runs_conceded,
            CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END     AS legal_ball,
            CASE WHEN be.wides = 0 AND be.no_balls = 0
                      AND COALESCE(be.runs,0)=0
                      AND COALESCE(be.byes,0)=0
                      AND COALESCE(be.leg_byes,0)=0
                      AND COALESCE(be.penalty_runs,0)=0
                 THEN 1 ELSE 0 END                                           AS dot_ball,
            CASE WHEN be.runs >= 4 THEN 1 ELSE 0 END                         AS boundary_ball,
            CASE
              WHEN be.dismissed_player_id = be.batter_id
               AND LOWER(COALESCE(TRIM(be.dismissal_type), '')) NOT IN
                   ('', 'not out', 'retired hurt', 'retired out', 'obstructing the field')
              THEN 1 ELSE 0
            END                                                               AS wicket_ball
          FROM ball_events be
          WHERE be.bowler_id IN ({in_placeholders})
        ),
        agg AS (
          SELECT
            phase,
            SUM(legal_ball)                                                   AS legal_balls,
            COUNT(*)                                                          AS total_balls,
            SUM(runs_conceded)                                                AS runs_conceded,
            SUM(dot_ball)                                                     AS dot_balls,
            SUM(boundary_ball)                                                AS boundary_balls,
            SUM(wicket_ball)                                                  AS wickets
          FROM raw
          GROUP BY phase
        )
        SELECT
          phase,
          ROUND(legal_balls / 6.0, 1)                                        AS overs,
          CASE WHEN legal_balls > 0 THEN ROUND(runs_conceded * 6.0 / legal_balls, 2) ELSE NULL END AS economy,
          ROUND(dot_balls * 100.0 / NULLIF(total_balls, 0), 1)               AS dot_pct,
          ROUND(wickets * 1.0 / NULLIF(total_balls, 0), 4)                    AS wickets_perc_ball,
          ROUND(boundary_balls * 100.0 / NULLIF(total_balls, 0), 1)          AS boundary_pct,
          total_balls
        FROM agg
        WHERE total_balls >= ?
        ORDER BY CASE phase
                   WHEN 'Powerplay' THEN 1
                   WHEN 'Middle'    THEN 2
                   WHEN 'Death'     THEN 3
                   ELSE 4
                 END
    """, (*opp_ids, payload.min_balls_bowling))
    bowling_by_phase_rows = [dict(r) for r in c.fetchall()]

    # by style (opponent bowlers' own style)
    c.execute(f"""
        WITH raw AS (
          SELECT
            {STYLE_CASE_SQL}                                                 AS style_norm,
            be.runs + be.wides + be.no_balls                                 AS runs_conceded,
            CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END     AS legal_ball,
            CASE WHEN be.wides = 0 AND be.no_balls = 0
                      AND COALESCE(be.runs,0)=0
                      AND COALESCE(be.byes,0)=0
                      AND COALESCE(be.leg_byes,0)=0
                      AND COALESCE(be.penalty_runs,0)=0
                 THEN 1 ELSE 0 END                                           AS dot_ball,
            CASE WHEN be.runs >= 4 THEN 1 ELSE 0 END                         AS boundary_ball,
            CASE
              WHEN be.dismissed_player_id = be.batter_id
               AND LOWER(COALESCE(TRIM(be.dismissal_type), '')) NOT IN
                   ('', 'not out', 'retired hurt', 'retired out', 'obstructing the field')
              THEN 1 ELSE 0
            END                                                               AS wicket_ball
          FROM ball_events be
          JOIN players bowl ON be.bowler_id = bowl.player_id
          WHERE be.bowler_id IN ({in_placeholders})
        ),
        agg AS (
          SELECT
            style_norm,
            SUM(legal_ball)                                                   AS legal_balls,
            COUNT(*)                                                          AS total_balls,
            SUM(runs_conceded)                                                AS runs_conceded,
            SUM(dot_ball)                                                     AS dot_balls,
            SUM(boundary_ball)                                                AS boundary_balls,
            SUM(wicket_ball)                                                  AS wickets
          FROM raw
          GROUP BY style_norm
        )
        SELECT
          style_norm,
          ROUND(legal_balls / 6.0, 1)                                        AS overs,
          CASE WHEN legal_balls > 0 THEN ROUND(runs_conceded * 6.0 / legal_balls, 2) ELSE NULL END AS economy,
          ROUND(dot_balls * 100.0 / NULLIF(total_balls, 0), 1)               AS dot_pct,
          ROUND(wickets * 1.0 / NULLIF(total_balls, 0), 4)                    AS wickets_perc_ball,
          ROUND(boundary_balls * 100.0 / NULLIF(total_balls, 0), 1)          AS boundary_pct,
          total_balls
        FROM agg
        WHERE total_balls >= ?
        ORDER BY style_norm
    """, (*opp_ids, payload.min_balls_bowling))
    bowling_by_style_rows = [dict(r) for r in c.fetchall()]

    # Bowling strengths/weaknesses (simple heuristics):
    strengths_bowl: List[str] = []
    weaknesses_bowl: List[str] = []

    if bowling_by_phase_rows:
        econ_sorted = sorted([r for r in bowling_by_phase_rows if r.get("economy") is not None],
                             key=lambda x: x["economy"])
        if econ_sorted:
            strengths_bowl += [f"Best economy in {econ_sorted[0]['phase']} (Eco {econ_sorted[0]['economy']})"]

        wk_sorted = sorted(bowling_by_phase_rows, key=lambda x: (x["wickets_perc_ball"] or 0), reverse=True)
        strengths_bowl += [f"Highest wicket threat in {wk_sorted[0]['phase']} ({wk_sorted[0]['wickets_perc_ball']}/ball)"]

        bp_sorted = sorted(bowling_by_phase_rows, key=lambda x: (x["boundary_pct"] or 0), reverse=True)
        weaknesses_bowl += [f"More boundaries in {bp_sorted[0]['phase']} ({bp_sorted[0]['boundary_pct']}%)"]

    if bowling_by_style_rows:
        econ_s = sorted([r for r in bowling_by_style_rows if r.get("economy") is not None],
                        key=lambda x: x["economy"])
        if econ_s:
            strengths_bowl += [f"Lowest economy with {econ_s[0]['style_norm']} (Eco {econ_s[0]['economy']})"]

        wk_s = sorted(bowling_by_style_rows, key=lambda x: (x["wickets_perc_ball"] or 0), reverse=True)
        strengths_bowl += [f"Most wickets/ball from {wk_s[0]['style_norm']} ({wk_s[0]['wickets_perc_ball']}/ball)"]

        bp_s = sorted(bowling_by_style_rows, key=lambda x: (x["boundary_pct"] or 0), reverse=True)
        weaknesses_bowl += [f"More boundaries vs {bp_s[0]['style_norm']} ({bp_s[0]['boundary_pct']}%)"]

    strengths_bowl = _dedup(strengths_bowl)[:4]
    weaknesses_bowl = _dedup(weaknesses_bowl)[:4]

    conn.close()

    return JSONResponse({
        "batting": {
            "strengths": strengths_bat,
            "weaknesses": weaknesses_bat,
            "by_style": batting_by_style,
            "by_phase": batting_by_phase,
        },
        "bowling": {
            "strengths": strengths_bowl,
            "weaknesses": weaknesses_bowl,
            "by_style": bowling_by_style_rows,
            "by_phase": bowling_by_phase_rows,
        }
    })

def _overs_from_legal_balls(legal_balls: int) -> float:
    return legal_balls / 6.0 if legal_balls else 0.0

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b else default
    except:
        return default

def _trimmed_median(values: List[float], trim: float = 0.1) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    k = int(n * trim)
    core = vals[k: n - k] if n - 2*k > 0 else vals
    return statistics.median(core)

def _fetchone(query: str, params: tuple = ()):
    rows = _fetchall(query, params)
    return rows[0] if rows else None

def _fetchall(query: str, params: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

@app.get("/batting-targets-advanced")
def batting_targets_advanced(
    team_category: str = Query(..., description="e.g., 'Women', 'Men', 'U19 Women'"),
    our_team: str = Query(..., description="e.g., 'Brasil Women'"),
    opponent_country: str = Query(..., description="e.g., 'Rwanda Women'"),
    ground: str = Query(..., description="Pure ground name (no time)"),
    time_of_day: Optional[str] = Query(None, description="If venue strings like 'Ground, Evening' exist"),
    recency_days: int = Query(720, description="Lookback (days) for stats weighting, default ~24 months"),
    include_rain: bool = Query(False, description="Include rain-interrupted/DLS matches in venue baseline?")
) -> Dict[str, Any]:
    """
    Returns:
      {
        venue: {...},
        par: { venue_par, adjusted_par, target_total },
        phases: [{phase, overs, runs, rpo}, ...],
        notes: [ ... ],
        debug: { ... }  # optional—hide in UI if you want
      }
    """

    # --------- Common filters ---------
    cat_like = f"%{team_category.lower()}%"
    cutoff = (datetime.utcnow() - timedelta(days=recency_days)).strftime("%Y-%m-%d")

    # Venue WHERE
    if time_of_day:
        venue_exact = f"{ground}, {time_of_day}"
        venue_where = "m.venue = ?"
        venue_params = (venue_exact,)
    else:
        # accept either exact ground or strings starting with "ground, ..."
        venue_where = " (m.venue = ? OR m.venue LIKE ?) "
        venue_params = (ground, f"{ground}, %")

    # Rain filter
    rain_where = "" if include_rain else " AND (m.rain_interrupted = 0 OR m.rain_interrupted IS NULL) "

    # --------- 1) VENUE BASELINE (first-innings, normalized to 20 overs, trimmed median) ---------
    # Pull first-innings totals + overs_bowled; normalize to 20 overs (RPO * 20), cap scaler to avoid huge jumps.
    venue_rows = _fetchall(f"""
        SELECT i.total_runs, i.overs_bowled
        FROM innings i
        JOIN matches m ON m.match_id = i.match_id
        WHERE i.innings = 1
          AND {venue_where}
          AND (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
          AND (m.match_date IS NULL OR m.match_date >= ?)
          {rain_where}
    """, (*venue_params, cat_like, cat_like, cutoff))

    eq20_samples = []
    for r in venue_rows:
        total = r["total_runs"] or 0
        overs = r["overs_bowled"] or 0.0
        if overs and overs > 0:
            rpo = total / overs
            # cap scaling factor to reduce extreme inflation/deflation
            eq20 = rpo * 20.0
            eq20_samples.append(eq20)
        else:
            eq20_samples.append(float(total))

    venue_par = _trimmed_median(eq20_samples, trim=0.1) if eq20_samples else None

    # As fallback, use league-wide baseline (same category) if venue has no data
    if venue_par is None:
        league_rows = _fetchall(f"""
            SELECT i.total_runs, i.overs_bowled
            FROM innings i
            JOIN matches m ON m.match_id = i.match_id
            WHERE i.innings = 1
              AND (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
              AND (m.match_date IS NULL OR m.match_date >= ?)
        """, (cat_like, cat_like, cutoff))
        league_eq20 = []
        for r in league_rows:
            total = r["total_runs"] or 0
            overs = r["overs_bowled"] or 0.0
            eq20 = (total / overs) * 20.0 if overs else float(total)
            league_eq20.append(eq20)
        venue_par = _trimmed_median(league_eq20, 0.1) if league_eq20 else 140.0  # sensible default if empty

    # --------- 2) OPPONENT BOWLING & LEAGUE BOWLING (economy) ---------
    # Opponent bowling, all balls where opponent is the bowling team (category filter on team strings)
    opp_bowl = _fetchone(f"""
        SELECT 
          SUM(COALESCE(be.runs,0) + COALESCE(be.wides,0) + COALESCE(be.no_balls,0)
              + COALESCE(be.byes,0) + COALESCE(be.leg_byes,0) + COALESCE(be.penalty_runs,0)) AS runs_conceded,
          SUM(CASE WHEN be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS legal_balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON m.match_id = i.match_id
        WHERE i.bowling_team = ?
          AND (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
          AND (m.match_date IS NULL OR m.match_date >= ?)
    """, (opponent_country, cat_like, cat_like, cutoff))

    opp_runs = (opp_bowl["runs_conceded"] or 0) if opp_bowl else 0
    opp_balls = (opp_bowl["legal_balls"] or 0) if opp_bowl else 0
    opp_econ = 6.0 * _safe_div(opp_runs, opp_balls, default=0.0)

    # League bowling economy across category
    league_bowl = _fetchone(f"""
        SELECT 
          SUM(COALESCE(be.runs,0) + COALESCE(be.wides,0) + COALESCE(be.no_balls,0)
              + COALESCE(be.byes,0) + COALESCE(be.leg_byes,0) + COALESCE(be.penalty_runs,0)) AS runs_conceded,
          SUM(CASE WHEN be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS legal_balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON m.match_id = i.match_id
        WHERE (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
          AND (m.match_date IS NULL OR m.match_date >= ?)
    """, (cat_like, cat_like, cutoff))

    lg_runs = (league_bowl["runs_conceded"] or 0) if league_bowl else 0
    lg_balls = (league_bowl["legal_balls"] or 0) if league_bowl else 0
    lg_econ = 6.0 * _safe_div(lg_runs, lg_balls, default=7.8)

    # Opponent phase economies (PP/Middle/Death) vs league
    def _phase_econ(where_flag: str, team: Optional[str]) -> float:
        # team=None => league
        rows = _fetchone(f"""
            SELECT 
              SUM(COALESCE(be.runs,0) + COALESCE(be.wides,0) + COALESCE(be.no_balls,0)
                  + COALESCE(be.byes,0) + COALESCE(be.leg_byes,0) + COALESCE(be.penalty_runs,0)) AS runs_conceded,
              SUM(CASE WHEN be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS legal_balls
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON m.match_id = i.match_id
            WHERE {where_flag} = 1
              {"AND i.bowling_team = ?" if team else ""}
              AND (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
              AND (m.match_date IS NULL OR m.match_date >= ?)
        """, ((team, cat_like, cat_like, cutoff) if team else (cat_like, cat_like, cutoff)))
        r = (rows["runs_conceded"] or 0) if rows else 0
        b = (rows["legal_balls"] or 0) if rows else 0
        return 6.0 * _safe_div(r, b, default=lg_econ)

    opp_pp_econ    = _phase_econ("be.is_powerplay", opponent_country)
    opp_mid_econ   = _phase_econ("be.is_middle_overs", opponent_country)
    opp_death_econ = _phase_econ("be.is_death_overs", opponent_country)

    lg_pp_econ    = _phase_econ("be.is_powerplay", None)
    lg_mid_econ   = _phase_econ("be.is_middle_overs", None)
    lg_death_econ = _phase_econ("be.is_death_overs", None)

    # --------- 3) OUR BATTING vs LEAGUE BATTING (RPO) ---------
    our_bat = _fetchone(f"""
        SELECT 
          SUM(COALESCE(be.runs,0) + COALESCE(be.wides,0) + COALESCE(be.no_balls,0)
              + COALESCE(be.byes,0) + COALESCE(be.leg_byes,0) + COALESCE(be.penalty_runs,0)) AS runs_scored,
          SUM(CASE WHEN be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS legal_balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON m.match_id = i.match_id
        WHERE i.batting_team = ?
          AND (LOWER(i.batting_team) LIKE ? OR LOWER(i.bowling_team) LIKE ?)
          AND (m.match_date IS NULL OR m.match_date >= ?)
    """, (our_team, cat_like, cat_like, cutoff))

    our_runs = (our_bat["runs_scored"] or 0) if our_bat else 0
    our_balls = (our_bat["legal_balls"] or 0) if our_bat else 0
    our_rpo = _safe_div(our_runs, _overs_from_legal_balls(our_balls), default=7.8)

    # League batting RPO (same category)
    # (Note: mathematically this equals lg_econ if you include byes/leg-byes symmetrically)
    lg_bat_rpo = _safe_div(lg_runs, _overs_from_legal_balls(lg_balls), default=7.8)

    # --------- 4) Adjust venue par by strengths (bounded ±15%) ---------
    # Opponent "weakness factor" in bowling (econ higher than league => >1 => weaker)
    opp_weakness = _safe_div(opp_econ, lg_econ, default=1.0)
    # Our batting strength vs league (rpo higher => >1)
    our_bat_strength = _safe_div(our_rpo, lg_bat_rpo, default=1.0)

    combined = our_bat_strength * opp_weakness
    adj_factor = max(min(combined - 1.0, 0.15), -0.15)  # clamp [-15%, +15%]

    adjusted_par = venue_par * (1.0 + adj_factor)
    target_total = math.ceil(adjusted_par * 1.10)  # +10% safety buffer

    # --------- 5) Phase allocation (shift shares by relative opp weakness per phase) ---------
    base_shares = {
        "Powerplay": 0.28,
        "Middle": 0.34,
        "Death": 0.38
    }
    # multipliers >1 => we push more weight into that phase
    mul_pp   = _safe_div(lg_pp_econ,   opp_pp_econ,   default=1.0)
    mul_mid  = _safe_div(lg_mid_econ,  opp_mid_econ,  default=1.0)
    mul_death= _safe_div(lg_death_econ,opp_death_econ,default=1.0)

    # Apply gentle shift (avoid extreme reallocation)
    def _limit_share(base, m):
        # turn multiplier into a +/- up to 5% shift:
        shift = max(min((m - 1.0) * 0.10, 0.05), -0.05)
        return base + shift

    share_pp   = _limit_share(base_shares["Powerplay"], mul_pp)
    share_death= _limit_share(base_shares["Death"], mul_death)
    # middle gets the remainder to sum to 1
    share_mid  = 1.0 - (share_pp + share_death)
    # guard rails
    if share_mid < 0.25:  # don't starve the middle
        deficit = 0.25 - share_mid
        share_mid += deficit
        # take evenly from others
        share_pp   -= deficit/2
        share_death-= deficit/2

    # Overs per phase
    overs_pp, overs_mid, overs_death = 6, 8, 6

    phases = [
        {
            "phase": "Powerplay 0–6",
            "overs": overs_pp,
            "runs": round(target_total * share_pp),
            "rpo": round(_safe_div(target_total * share_pp, overs_pp, 0), 2)
        },
        {
            "phase": "Middle 7–14",
            "overs": overs_mid,
            "runs": round(target_total * share_mid),
            "rpo": round(_safe_div(target_total * share_mid, overs_mid, 0), 2)
        },
        {
            "phase": "Death 15–20",
            "overs": overs_death,
            "runs": round(target_total * share_death),
            "rpo": round(_safe_div(target_total * share_death, overs_death, 0), 2)
        },
    ]

    # Optional notes
    notes = []
    if opp_pp_econ > lg_pp_econ:
        notes.append("Opposition PP economy above global normal → license to score early.")
    if opp_death_econ > lg_death_econ:
        notes.append("Opposition Death economy above global normal → back-end acceleration viable.")

    return {
        "venue": {
            "ground": ground,
            "time_of_day": time_of_day or "",
            "cutoff_since": cutoff,
            "samples_used": len(eq20_samples),
        },
        "par": {
            "venue_par": round(venue_par, 1) if venue_par is not None else None,
            "adjusted_par": round(adjusted_par, 1),
            "target_total": int(target_total),
            "safety_buffer_pct": 10
        },
        "phases": phases,
        "notes": notes,
        "debug": {
            "opp_econ": round(opp_econ, 2),
            "lg_econ": round(lg_econ, 2),
            "our_rpo": round(our_rpo, 2),
            "lg_bat_rpo": round(lg_bat_rpo, 2),
            "phase_econ": {
                "opp": {"pp": round(opp_pp_econ,2), "mid": round(opp_mid_econ,2), "death": round(opp_death_econ,2)},
                "lg":  {"pp": round(lg_pp_econ,2),  "mid": round(lg_mid_econ,2),  "death": round(lg_death_econ,2)},
            },
            "shares": {
                "pp": round(share_pp,3),
                "mid": round(share_mid,3),
                "death": round(share_death,3),
            }
        }
    }

def _safe_num(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def _empirical_bayes_rate(successes: float, trials: float, prior_mean: float, prior_weight: float):
    """
    Simple EB shrinkage for rates (e.g., SR per ball = runs*100 / balls).
    successes/trials should be scaled to the target rate you want.
    """
    t = max(_safe_num(trials, 0.0), 0.0)
    if t <= 0.0:
        return _safe_num(prior_mean, 0.0)
    return ( _safe_num(successes,0.0) + _safe_num(prior_weight,0.0) * _safe_num(prior_mean,0.0) ) / (t + _safe_num(prior_weight,0.0))

def _z_score(val: float, mean: float, sd: float):
    v = _safe_num(val, None); m = _safe_num(mean, None); s = _safe_num(sd, None)
    if v is None or m is None or s is None or s <= 1e-9:
        return 0.0
    return (v - m) / s

def _confidence_from_sample(n) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    if n >= 240: return "High"
    if n >= 120: return "Medium-High"
    if n >= 60:  return "Medium"
    if n >= 30:  return "Low"
    return "Very Low"

def _resolve_conflicts(items, key_fn):
    """
    Keep only highest-scoring item per topic key.
    """
    best = {}
    for it in items:
        k = key_fn(it)
        if k not in best or _safe_num(it.get("score", 0.0)) > _safe_num(best[k].get("score", 0.0)):
            best[k] = it
    return list(best.values())

def _style_norm_case_sql() -> str:
    # Normalize bowler style using style + arm into compact buckets you use elsewhere
    return """
        CASE
            WHEN bowl.bowling_style = 'Off Spin' AND bowl.bowling_arm = 'Left'  THEN 'SLA'
            WHEN bowl.bowling_style = 'Off Spin' AND bowl.bowling_arm = 'Right' THEN 'Off Spin'
            WHEN bowl.bowling_style = 'Leg Spin'  AND bowl.bowling_arm = 'Left'  THEN 'Left Wrist'
            ELSE bowl.bowling_style
        END
    """

def _length_case_sql(py_col: str = "be.pitch_y") -> str:
    return f"""
        CASE
            WHEN {py_col} IS NULL THEN 'Unknown'
            WHEN {py_col} < 0.03636 THEN 'Full Toss'
            WHEN {py_col} < 0.1636  THEN 'Yorker'
            WHEN {py_col} < 0.31818 THEN 'Full'
            WHEN {py_col} < 0.545454 THEN 'Good'
            ELSE 'Short'
        END
    """

def _line_case_sql(px_col: str = "be.pitch_x") -> str:
    return f"""
        CASE
            WHEN {px_col} IS NULL THEN 'Unknown'
            WHEN {px_col} <= 0.26 THEN 'Wide Outside Off'
            WHEN {px_col} <= 0.44 THEN 'Outside Off'
            WHEN {px_col} <= 0.55 THEN 'Middle/Off'
            ELSE 'Leg'
        END
    """


@app.post("/do-donts")
def do_donts(payload: Dict[str, Any] = Body(...)):
    """
    Produces data-driven Do & Don’ts across phases, both batting & bowling.
    Request JSON (keys used):
      team_category (optional)
      our_team (required)
      opponent_country (required)
      ground (optional), time_of_day (optional)
      min_balls_by_style (default 60)
      min_balls_death_phase (default 36)

    Response JSON shape matches your PreGame UI expectations.
    """
    OUR_TEAM   = payload.get("our_team")
    OPP_TEAM   = payload.get("opponent_country")
    if not OUR_TEAM or not OPP_TEAM:
        return {
            "context": {},
            "batting": {"do": [], "dont": []},
            "bowling": {"do": [], "dont": []}
        }

    GROUND     = payload.get("ground")
    TOD        = payload.get("time_of_day")

    MIN_STYLE  = int(payload.get("min_balls_by_style", 60) or 60)
    MIN_DEATH  = int(payload.get("min_balls_death_phase", 36) or 36)

    # Build venue filter values: m.venue exact or "ground, time"
    venue_filter_values: List[str] = []
    if GROUND and TOD:
        venue_filter_values = [f"{GROUND}, {TOD}"]

    # SQL helpers
    style_norm = _style_norm_case_sql()
    length_case = _length_case_sql()
    line_case   = _line_case_sql()

    conn = _db()
    c = conn.cursor()

    # ---------------- Batting: DO & DON'T by bowler style per phase ----------------
    PHASES = [
        ("Powerplay",    "be.is_powerplay = 1",     0.9),
        ("Middle Overs", "be.is_middle_overs = 1",  1.0),
        ("Death Overs",  "be.is_death_overs = 1",   1.2),
    ]

    batting_do: List[Dict[str, Any]] = []
    batting_dont: List[Dict[str, Any]] = []

    for phase_name, phase_cond, phase_wt in PHASES:
        # Opponent batting vs style
        c.execute(f"""
            WITH balls AS (
                SELECT
                    {style_norm} AS style_norm,
                    1 AS ball,
                    be.runs AS runs,
                    CASE WHEN be.runs=0 THEN 1 ELSE 0 END AS dot,
                    CASE WHEN be.runs>=4 THEN 1 ELSE 0 END AS boundary,
                    CASE WHEN be.dismissal_type IS NOT NULL
                         AND LOWER(be.dismissal_type) NOT IN ('run out','retired out','retired hurt','obstructing the field','not out')
                         THEN 1 ELSE 0 END AS out
                FROM ball_events be
                JOIN innings i ON be.innings_id=i.innings_id
                JOIN matches m ON i.match_id=m.match_id
                JOIN players bowl ON be.bowler_id=bowl.player_id
                WHERE i.batting_team = ? AND {phase_cond} AND be.wides=0
                  { " AND (m.venue = ? OR m.venue LIKE ?)" if (GROUND and not TOD) else "" }
                  { f" AND m.venue IN ({','.join('?'*len(venue_filter_values))})" if len(venue_filter_values)>0 else "" }
            ),
            agg AS (
                SELECT
                    style_norm,
                    SUM(ball) AS balls,
                    SUM(runs) AS runs,
                    SUM(dot) AS dots,
                    SUM(boundary) AS boundaries,
                    SUM(out) AS outs
                FROM balls GROUP BY style_norm
            )
            SELECT
                style_norm, balls, runs, dots, boundaries, outs,
                ROUND(CASE WHEN balls>0 THEN (runs*100.0)/balls END,1) AS sr
            FROM agg
            ORDER BY style_norm
        """, ([OPP_TEAM] +
              ([GROUND, f"{GROUND}, %"] if (GROUND and not TOD) else []) +
              (venue_filter_values if len(venue_filter_values)>0 else [])
        ))
        opp_rows = {r["style_norm"]: dict(r) for r in c.fetchall()}

        # Venue baseline: everyone at same filters
        c.execute(f"""
            WITH balls AS (
                SELECT
                    {style_norm} AS style_norm,
                    1 AS ball,
                    be.runs AS runs
                FROM ball_events be
                JOIN innings i ON be.innings_id=i.innings_id
                JOIN matches m ON i.match_id=m.match_id
                JOIN players bowl ON be.bowler_id=bowl.player_id
                WHERE {phase_cond} AND be.wides=0
                  { " AND (m.venue = ? OR m.venue LIKE ?)" if (GROUND and not TOD) else "" }
                  { f" AND m.venue IN ({','.join('?'*len(venue_filter_values))})" if len(venue_filter_values)>0 else "" }
            ),
            base AS (
                SELECT
                    style_norm,
                    SUM(ball) AS balls,
                    SUM(runs) AS runs
                FROM balls GROUP BY style_norm
            )
            SELECT
                style_norm,
                balls,
                CASE WHEN balls>0 THEN (runs*100.0)/balls END AS sr
            FROM base
        """, (([GROUND, f"{GROUND}, %"] if (GROUND and not TOD) else []) +
              (venue_filter_values if len(venue_filter_values)>0 else [])
        ))
        base_rows = {r["style_norm"]: dict(r) for r in c.fetchall()}

        # score each style
        items = []
        for style, r in opp_rows.items():
            balls = _safe_num(r.get("balls"), 0)
            if balls <= 0: 
                continue
            base = base_rows.get(style, {})
            base_sr = _safe_num(base.get("sr"), 0.0)

            # EB shrinkage on SR (successes = runs*100, trials = balls; prior = baseline SR, weight ~ 90)
            shr_sr = _empirical_bayes_rate(_safe_num(r.get("runs"),0.0)*100.0, balls, base_sr, 90.0)
            delta  = shr_sr - base_sr
            z      = _z_score(shr_sr, base_sr, 8.0)  # conservative SR sd ~8

            score = (0.6*z + 0.4*(delta/10.0)) * phase_wt

            items.append({
                "style": style,
                "phase": phase_name,
                "balls": int(balls),
                "shr_sr": round(shr_sr,1),
                "base_sr": round(base_sr,1),
                "delta_sr": round(delta,1),
                "score": score
            })

        POS_MIN = MIN_STYLE
        pos = [x for x in items if x["score"] > 0 and x["balls"] >= POS_MIN]
        neg = [x for x in items if x["score"] < 0 and x["balls"] >= POS_MIN]

        if pos:
            best = max(pos, key=lambda x: x["score"])
            batting_do.append({
                "text": f"In the {best['phase']}, lean into scoring vs {best['style']}.",
                "phase": best["phase"],
                "evidence": f"Shrunk SR {best['shr_sr']} vs venue {best['base_sr']} (Δ {best['delta_sr']}), balls {best['balls']}.",
                "confidence": _confidence_from_sample(best["balls"]),
                "score": best["score"]
            })
        if neg:
            worst = min(neg, key=lambda x: x["score"])
            batting_dont.append({
                "text": f"Avoid prolonged exposure to {worst['style']} in the {worst['phase']}.",
                "phase": worst["phase"],
                "evidence": f"Shrunk SR {worst['shr_sr']} below venue {worst['base_sr']} (Δ {worst['delta_sr']}), balls {worst['balls']}.",
                "confidence": _confidence_from_sample(worst["balls"]),
                "score": worst["score"]
            })

    # ---------------- Bowling: Middle-overs DO by style; Death-pocket DON'T ----------------
    bowling_do: List[Dict[str, Any]] = []
    bowling_dont: List[Dict[str, Any]] = []

    # Middle overs team-level (our bowling vs their batting) – reuse your earlier logic & EB score
    # Opponent batting is the same; we flip interpretation: a *lower* opponent SR vs a style is good for our bowling.
    c.execute(f"""
        WITH balls AS (
            SELECT
                {style_norm} AS style_norm,
                1 AS ball,
                be.runs AS runs
            FROM ball_events be
            JOIN innings i ON be.innings_id=i.innings_id
            JOIN matches m ON i.match_id=m.match_id
            JOIN players bowl ON be.bowler_id=bowl.player_id
            WHERE i.batting_team = ? AND be.is_middle_overs = 1 AND be.wides=0
              { " AND (m.venue = ? OR m.venue LIKE ?)" if (GROUND and not TOD) else "" }
              { f" AND m.venue IN ({','.join('?'*len(venue_filter_values))})" if len(venue_filter_values)>0 else "" }
        ),
        agg AS (
            SELECT style_norm, SUM(ball) AS balls, SUM(runs) AS runs
            FROM balls GROUP BY style_norm
        )
        SELECT style_norm, balls, runs, 
               ROUND(CASE WHEN balls>0 THEN (runs*100.0)/balls END,1) AS sr
        FROM agg
    """, ([OPP_TEAM] +
          ([GROUND, f"{GROUND}, %"] if (GROUND and not TOD) else []) +
          (venue_filter_values if len(venue_filter_values)>0 else [])
    ))
    opp_mo = {r["style_norm"]: dict(r) for r in c.fetchall()}

    c.execute(f"""
        WITH balls AS (
            SELECT
                {style_norm} AS style_norm,
                1 AS ball,
                be.runs AS runs
            FROM ball_events be
            JOIN innings i ON be.innings_id=i.innings_id
            JOIN matches m ON i.match_id=m.match_id
            JOIN players bowl ON be.bowler_id=bowl.player_id
            WHERE be.is_middle_overs = 1 AND be.wides=0
              { " AND (m.venue = ? OR m.venue LIKE ?)" if (GROUND and not TOD) else "" }
              { f" AND m.venue IN ({','.join('?'*len(venue_filter_values))})" if len(venue_filter_values)>0 else "" }
        ),
        base AS (
            SELECT style_norm, SUM(ball) AS balls, SUM(runs) AS runs
            FROM balls GROUP BY style_norm
        )
        SELECT style_norm, balls, CASE WHEN balls>0 THEN (runs*100.0)/balls END AS sr
        FROM base
    """, (([GROUND, f"{GROUND}, %"] if (GROUND and not TOD) else []) +
          (venue_filter_values if len(venue_filter_values)>0 else [])
    ))
    base_mo = {r["style_norm"]: dict(r) for r in c.fetchall()}

    mo_items = []
    for style, r in opp_mo.items():
        balls = _safe_num(r.get("balls"), 0)
        if balls <= 0: continue
        base_sr = _safe_num(base_mo.get(style, {}).get("sr"), 0.0)
        shr_sr  = _empirical_bayes_rate(_safe_num(r.get("runs"),0.0)*100.0, balls, base_sr, 90.0)
        delta   = base_sr - shr_sr   # lower than baseline = good for our bowling
        z       = _z_score(base_sr - shr_sr, 0.0, 8.0)  # positive if we suppress better than baseline

        score = (0.6*z + 0.4*(delta/10.0)) * 1.0  # phase weight 1.0 for MO
        mo_items.append({
            "style": style,
            "phase": "Middle Overs",
            "balls": int(balls),
            "shr_opp_sr": round(shr_sr,1),
            "venue_sr": round(base_sr,1),
            "delta_sr_suppression": round(delta,1),
            "score": score
        })

    mo_pos = [x for x in mo_items if x["balls"] >= MIN_STYLE and x["score"] > 0]
    if mo_pos:
        best = max(mo_pos, key=lambda x: x["score"])
        bowling_do.append({
            "text": f"Use {best['style']} through the middle with control fields.",
            "phase": best["phase"],
            "evidence": f"Opp SR shrunk {best['shr_opp_sr']} vs venue {best['venue_sr']} (suppression +{best['delta_sr_suppression']}), balls {best['balls']}.",
            "confidence": _confidence_from_sample(best["balls"]),
            "score": best["score"]
        })

    # Death pocket: DON'T vs finisher hot-zone (seam back-of-length on Leg)
    # Find best death batter by SR with enough balls
    c.execute(f"""
        WITH death_balls AS (
            SELECT
                bat.player_id, bat.player_name,
                1 AS ball, be.runs AS runs
            FROM ball_events be
            JOIN innings i ON be.innings_id=i.innings_id
            JOIN matches m ON i.match_id=m.match_id
            JOIN players bat ON be.batter_id=bat.player_id
            WHERE be.is_death_overs=1 AND be.wides=0
              AND i.batting_team = ?
              { " AND (m.venue = ? OR m.venue LIKE ?)" if (GROUND and not TOD) else "" }
              { f" AND m.venue IN ({','.join('?'*len(venue_filter_values))})" if len(venue_filter_values)>0 else "" }
        ),
        agg AS (
            SELECT player_id, player_name,
                   SUM(ball) AS balls,
                   SUM(runs) AS runs,
                   ROUND(CASE WHEN SUM(ball)>0 THEN (SUM(runs)*100.0)/SUM(ball) END,1) AS sr
            FROM death_balls GROUP BY player_id, player_name
        )
        SELECT * FROM agg WHERE balls >= ? ORDER BY sr DESC, balls DESC LIMIT 1
    """, ([OPP_TEAM] +
          ([GROUND, f"{GROUND}, %"] if (GROUND and not TOD) else []) +
          (venue_filter_values if len(venue_filter_values)>0 else []) +
          [MIN_DEATH]
    ))
    finisher = c.fetchone()
    if finisher:
        finisher = dict(finisher)
        finisher_id = finisher["player_id"]

        # Finisher hot pocket: seam (Pace/Medium) & (Good/Short) & line Leg @ death
        where_pocket = [
            "be.is_death_overs=1",
            "be.wides=0",
            "be.batter_id = ?",
            "bowl.player_id = be.bowler_id",
            "bowl.bowling_style IN ('Pace','Medium')",
            f"{length_case} IN ('Good','Short')",
            f"{line_case} = 'Leg'"
        ]
        params_pocket: List[Any] = [finisher_id]

        if GROUND and not TOD:
            where_pocket.append("(m.venue = ? OR m.venue LIKE ?)")
            params_pocket.extend([GROUND, f"{GROUND}, %"])
        elif len(venue_filter_values) > 0:
            where_pocket.append("m.venue IN ({})".format(",".join("?"*len(venue_filter_values))))
            params_pocket.extend(venue_filter_values)

        c.execute(f"""
            WITH pocket AS (
                SELECT 1 AS ball, be.runs AS runs
                FROM ball_events be
                JOIN innings i ON be.innings_id=i.innings_id
                JOIN matches m ON i.match_id=m.match_id
                JOIN players bowl ON be.bowler_id=bowl.player_id
                WHERE {" AND ".join(where_pocket)}
            ),
            agg AS (
                SELECT
                    SUM(ball) AS balls,
                    SUM(runs) AS runs,
                    ROUND(CASE WHEN SUM(ball)>0 THEN SUM(runs)*1.0/SUM(ball) END,2) AS rpb,
                    ROUND(CASE WHEN SUM(ball)>0 THEN (SUM(runs)*100.0)/SUM(ball) END,1) AS sr,
                    ROUND(SUM(CASE WHEN runs>=4 THEN 1 ELSE 0 END)*100.0/NULLIF(SUM(ball),0),1) AS boundary_pct,
                    ROUND(SUM(CASE WHEN runs=0 THEN 1 ELSE 0 END)*100.0/NULLIF(SUM(ball),0),1) AS dot_pct
                FROM pocket
            )
            SELECT * FROM agg
        """, params_pocket)
        pocket = dict(c.fetchone() or {})

        # Venue baseline for same pocket (all batters)
        where_base = [
            "be.is_death_overs=1",
            "be.wides=0",
            "bowl.player_id = be.bowler_id",
            "bowl.bowling_style IN ('Pace','Medium')",
            f"{length_case} IN ('Good','Short')",
            f"{line_case} = 'Leg'"
        ]
        params_base: List[Any] = []
        if GROUND and not TOD:
            where_base.append("(m.venue = ? OR m.venue LIKE ?)")
            params_base.extend([GROUND, f"{GROUND}, %"])
        elif len(venue_filter_values) > 0:
            where_base.append("m.venue IN ({})".format(",".join("?"*len(venue_filter_values))))
            params_base.extend(venue_filter_values)

        c.execute(f"""
            WITH pocket AS (
                SELECT 1 AS ball, be.runs AS runs
                FROM ball_events be
                JOIN innings i ON be.innings_id=i.innings_id
                JOIN matches m ON i.match_id=m.match_id
                JOIN players bowl ON be.bowler_id=bowl.player_id
                WHERE {" AND ".join(where_base)}
            ),
            agg AS (
                SELECT
                    SUM(ball) AS balls,
                    SUM(runs) AS runs,
                    ROUND(CASE WHEN SUM(ball)>0 THEN SUM(runs)*1.0/SUM(ball) END,2) AS rpb,
                    ROUND(CASE WHEN SUM(ball)>0 THEN (SUM(runs)*100.0)/SUM(ball) END,1) AS sr
                FROM pocket
            )
            SELECT * FROM agg
        """, params_base)
        base_pocket = dict(c.fetchone() or {})

        p_balls = int(_safe_num(pocket.get("balls"), 0))
        if p_balls > 0:
            sr_gain = None
            if pocket.get("sr") is not None and base_pocket.get("sr") is not None:
                sr_gain = round(_safe_num(pocket["sr"]) - _safe_num(base_pocket["sr"]), 1)

            bowling_dont.append({
                "text": f"Don’t bowl back-of-length seam on leg at the death to {finisher['player_name']}.",
                "phase": "Death Overs",
                "evidence": f"Pocket SR {pocket.get('sr')} vs venue {base_pocket.get('sr')} (Δ {sr_gain}), balls {p_balls}.",
                "confidence": _confidence_from_sample(p_balls),
                "score": _safe_num(sr_gain, 0.0)  # larger gain = worse for us → stronger DON'T
            })

    # ---------------- Clean up: resolve contradictions & limit to top 3 ----------------
    batting_do   = _resolve_conflicts(batting_do,   key_fn=lambda x: (x.get("phase"), x.get("text")))
    batting_dont = _resolve_conflicts(batting_dont, key_fn=lambda x: (x.get("phase"), x.get("text")))
    bowling_do   = _resolve_conflicts(bowling_do,   key_fn=lambda x: (x.get("phase"), x.get("text")))
    bowling_dont = _resolve_conflicts(bowling_dont, key_fn=lambda x: (x.get("phase"), x.get("text")))

    batting_do   = sorted(batting_do,   key=lambda x: _safe_num(x.get("score"),0.0), reverse=True)[:3]
    batting_dont = sorted(batting_dont, key=lambda x: _safe_num(x.get("score"),0.0), reverse=True)[:3]
    bowling_do   = sorted(bowling_do,   key=lambda x: _safe_num(x.get("score"),0.0), reverse=True)[:3]
    bowling_dont = sorted(bowling_dont, key=lambda x: _safe_num(x.get("score"),0.0), reverse=True)[:3]

    # strip score from response
    for lst in (batting_do, batting_dont, bowling_do, bowling_dont):
        for it in lst:
            it.pop("score", None)

    context = {}
    if GROUND: context["ground"] = GROUND
    if TOD:    context["time_of_day"] = TOD

    conn.close()
    return {
        "context": context,
        "batting": {"do": batting_do, "dont": batting_dont},
        "bowling": {"do": bowling_do, "dont": bowling_dont}
    }

def get_country_stats(country, tournaments, selected_stats, selected_phases, bowler_type, bowling_arm, team_category, selected_matches=None):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ✅ Get country ID
    c.execute("SELECT country_id FROM countries WHERE country_name = ?", (country,))
    country_result = c.fetchone()
    if not country_result:
        return defaultdict(lambda: defaultdict(float))
    country_id = country_result[0]

    # ✅ Get tournament IDs
    c.execute(f"SELECT tournament_id FROM tournaments WHERE tournament_name IN ({','.join(['?']*len(tournaments))})", tournaments)
    tournament_ids = [row[0] for row in c.fetchall()]
    if not tournament_ids:
        return defaultdict(lambda: defaultdict(float))

    # ✅ Get matches for that country in those tournaments and matching the team category
    team_category_likes = [f"%{team_category}%", f"{team_category}%"]

    c.execute(f"""
        SELECT m.match_id
        FROM matches m
        JOIN countries c1 ON m.team_a = c1.country_id
        JOIN countries c2 ON m.team_b = c2.country_id
        WHERE m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})
        AND (m.team_a = ? OR m.team_b = ?)
        AND (
            c1.country_name LIKE ? OR c1.country_name LIKE ?
            OR c2.country_name LIKE ? OR c2.country_name LIKE ?
        )
    """, tournament_ids + [country_id, country_id] + team_category_likes * 2)

    match_ids = [row[0] for row in c.fetchall()]

    if not match_ids:
        print("❌ No matches found for country ID", country_id, "with teamCategory", team_category)
        return defaultdict(lambda: defaultdict(float))

    # Apply frontend match filter here if provided
    if selected_matches:
        filtered_match_ids = [m for m in match_ids if m in selected_matches]
        if not filtered_match_ids:
            print("❌ No matches after applying selectedMatches filter")
            return defaultdict(lambda: defaultdict(float))
        match_ids = filtered_match_ids

    # Build the match filter for SQL using the filtered match_ids
    match_filter = f"i.match_id IN ({','.join(['?'] * len(match_ids))})"


    # Bowler filters
    bowler_type_conditions = {
        "Pace": "p.bowling_style = 'Pace'",
        "Medium": "p.bowling_style = 'Medium'",
        "Leg Spin": "p.bowling_style = 'Leg Spin'",
        "Off Spin": "p.bowling_style = 'Off Spin'"
    }
    bowling_arm_conditions = {
        "Left": "p.bowling_arm = 'Left'",
        "Right": "p.bowling_arm = 'Right'"
    }
    
    type_clauses = [bowler_type_conditions[bt] for bt in bowler_type if bt in bowler_type_conditions]
    arm_clauses = [bowling_arm_conditions[arm] for arm in bowling_arm if arm in bowling_arm_conditions]

    combined_filter_parts = []
    if type_clauses:
        combined_filter_parts.append("(" + " OR ".join(type_clauses) + ")")
    if arm_clauses:
        combined_filter_parts.append("(" + " OR ".join(arm_clauses) + ")")
    combined_filter = " AND ".join(combined_filter_parts)

    # Match filter
    match_filter = f"i.match_id IN ({','.join(['?'] * len(match_ids))})"

    # Batter filter: batters must be from the country being analyzed
    batter_filter = "be.batter_id IN (SELECT player_id FROM players WHERE country_id = ?)"

    # Bowler filter for batting query (no country restriction, only type/arm filters)
    bowler_filter_batting = f"""
        be.bowler_id IN (
            SELECT p.player_id FROM players p
            WHERE 1=1 {' AND ' + combined_filter if combined_filter else ''}
        )
    """

    # Bowler filter for bowling query (must belong to country + type/arm filters)
    bowler_filter_bowling = f"""
        be.bowler_id IN (
            SELECT p.player_id FROM players p
            WHERE p.country_id = ? {' AND ' + combined_filter if combined_filter else ''}
        )
    """

    # Phase filter (powerplay, middle overs, death)
    phase_conditions = {
        'Powerplay': 'be.is_powerplay = 1',
        'Middle Overs': 'be.is_middle_overs = 1',
        'Death Overs': 'be.is_death_overs = 1'
    }
    phase_clauses = [phase_conditions[p] for p in selected_phases if p in phase_conditions]
    phase_filter = f"({' OR '.join(phase_clauses)})" if phase_clauses else "1=1"


    global_batting_conditions = f"{match_filter} AND {batter_filter} AND {bowler_filter_batting} AND {phase_filter}"
    global_bowling_conditions = f"{match_filter} AND {bowler_filter_bowling} AND {phase_filter}"
    fielder_filter = "p.country_id = ?"
    global_fielding_conditions = f"{match_filter} AND {phase_filter} AND {fielder_filter}"

    stats = defaultdict(lambda: defaultdict(float))

    # Batting query
    batting_query = f"""
        SELECT
            COUNT(DISTINCT be.batter_id || '-' || be.innings_id) AS batting_innings,
            SUM(be.runs) AS runs_off_bat,
            SUM(be.wides) + SUM(be.no_balls) + SUM(be.byes) + SUM(be.leg_byes) AS extras,
            SUM(CASE 
                WHEN be.wides = 0 THEN 1 ELSE 0
            END) AS balls_faced,
            SUM(be.dot_balls) AS dot_balls,
            SUM(CASE WHEN be.runs = 1 THEN 1 ELSE 0 END) AS ones,
            SUM(CASE WHEN be.runs = 2 THEN 1 ELSE 0 END) AS twos,
            SUM(CASE WHEN be.runs = 3 THEN 1 ELSE 0 END) AS threes,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            SUM(CASE 
                WHEN be.dismissed_player_id IN (SELECT player_id FROM players WHERE country_id = ?) 
                AND be.dismissal_type IS NOT NULL 
                AND LOWER(be.dismissal_type) != 'not out'
                THEN 1 ELSE 0 
            END) AS dismissals,
            SUM(CASE WHEN LOWER(be.shot_type) = 'attacking' THEN 1 ELSE 0 END) AS attacking,
            SUM(CASE WHEN LOWER(be.shot_type) = 'defensive' THEN 1 ELSE 0 END) AS defensive,
            SUM(CASE WHEN LOWER(be.shot_type) = 'rotation' THEN 1 ELSE 0 END) AS rotation,
            AVG(be.batting_intent_score) AS avg_intent
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE {global_batting_conditions}
    """



    stats = defaultdict(lambda: defaultdict(float))
    c.execute(batting_query, match_ids + [country_id, country_id])
    batting_data = c.fetchone()
    if batting_data:
        stats['batting']['Innings'] = batting_data[0] or 0
        stats['batting']['Runs Off Bat'] = batting_data[1] or 0
        stats['batting']['Batting Extras'] = batting_data[2] or 0
        stats['batting']['Total Runs'] = stats['batting']['Runs Off Bat'] + stats['batting']['Batting Extras']
        stats['batting']['Balls Faced'] = batting_data[3] or 0
        stats['batting']['Dot Balls Faced'] = batting_data[4] or 0
        stats['batting']['1s'] = batting_data[5] or 0
        stats['batting']['2s'] = batting_data[6] or 0
        stats['batting']['3s'] = batting_data[7] or 0
        stats['batting']['4s'] = batting_data[8] or 0
        stats['batting']['6s'] = batting_data[9] or 0
        stats['batting']['Dismissals'] = batting_data[10] or 0

        if stats['batting']['Balls Faced'] > 0:
            stats['batting']['Strike Rate'] = round(
                (stats['batting']['Runs Off Bat'] * 100 / stats['batting']['Balls Faced']), 2)
            stats['batting']['Scoring Shot %'] = round(
                (1 - (stats['batting']['Dot Balls Faced'] / stats['batting']['Balls Faced'])) * 100, 2)

        if stats['batting']['Dismissals'] > 0:
            stats['batting']['Batters Average'] = round(
                stats['batting']['Runs Off Bat'] / stats['batting']['Dismissals'], 2)

        total_intent = sum(filter(None, [batting_data[11], batting_data[12], batting_data[13]]))
        if total_intent > 0:
            stats['batting']['Attacking Shot %'] = round((batting_data[11] / total_intent) * 100, 2)
            stats['batting']['Defensive Shot %'] = round((batting_data[12] / total_intent) * 100, 2)
            stats['batting']['Rotation Shot %'] = round((batting_data[13] / total_intent) * 100, 2)

        if batting_data[14] is not None:
            stats['batting']['Avg Intent Score'] = round(batting_data[14], 2)


# Bowling
    bowling_query = f"""
    SELECT
        COUNT(*) AS total_balls,
        SUM(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END) AS legal_balls,
        SUM(be.runs) + SUM(be.wides) + SUM(be.no_balls) AS runs_conceded,
        SUM(CASE 
            WHEN be.dismissal_type IS NOT NULL 
            AND LOWER(be.dismissal_type) NOT IN ('run out', 'retired out', 'obstructing the field', 'retired not out') 
            AND be.dismissed_player_id = be.batter_id
            THEN 1 ELSE 0 
        END) AS wickets,
        SUM(CASE 
            WHEN be.runs = 0 AND be.wides = 0 AND be.no_balls = 0 
            THEN 1 ELSE 0 
        END) AS dot_balls,
        SUM(be.wides + be.no_balls) AS extras,
        SUM(CASE WHEN be.runs IN (4,6) THEN 1 ELSE 0 END) AS boundaries
    FROM ball_events be
    JOIN innings i ON be.innings_id = i.innings_id
    WHERE {global_bowling_conditions}
    """
    c.execute(bowling_query, match_ids + [country_id])
    bowling_data = c.fetchone()

    if bowling_data:
        total_balls = bowling_data[0] or 0
        legal_balls = bowling_data[1] or 0

        stats['bowling']['Overs'] = f"{legal_balls // 6}.{legal_balls % 6}"
        stats['bowling']['Runs Conceded'] = bowling_data[2]
        stats['bowling']['Wickets'] = bowling_data[3]
        stats['bowling']['Dot Balls Bowled'] = bowling_data[4]
        stats['bowling']['Extras'] = bowling_data[5]
        stats['bowling']['Boundaries Conceded'] = bowling_data[6]

        if bowling_data[0] > 0:
            stats['bowling']['Economy'] = round((bowling_data[2] / (legal_balls / 6)), 2)
            stats['bowling']['Dot Ball %'] = round(((bowling_data[4] / total_balls) * 100), 2)
            if bowling_data[3] > 0:
                stats['bowling']['Bowlers Average'] = round((bowling_data[2] / bowling_data[3]), 2)

    #Fielding

    fielding_weights = {
        'Taken Half Chance': 5,
        'Catch': 3,
        'Run Out': 3,
        'Direct Hit': 2,
        'Clean Stop/Pick Up': 1,
        'Boundary Save': 2,
        'Drop Catch': -3,
        'Missed Run Out': -2,
        'Missed Catch': -2,
        'Missed Fielding': -1,
        'Missed Half Chance': -0.5,
        'Fumble': -1,
        'Overthrow': -2
    }


    fielding_query = f"""
    SELECT
        SUM(CASE WHEN fe.event_name = 'Catch' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Run Out' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Drop Catch' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Boundary Save' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Clean Stop/Pick Up' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Direct Hit' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Missed Catch' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Missed Run Out' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Fumble' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Missed Fielding' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Overthrow' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Taken Half Chance' THEN 1 ELSE 0 END),
        SUM(CASE WHEN fe.event_name = 'Missed Half Chance' THEN 1 ELSE 0 END)
    FROM ball_fielding_events bfe
    JOIN fielding_events fe ON bfe.event_id = fe.event_id
    JOIN ball_events be ON bfe.ball_id = be.ball_id
    JOIN innings i ON be.innings_id = i.innings_id
    JOIN matches m ON i.match_id = m.match_id
    JOIN players p ON be.fielder_id = p.player_id
    WHERE {global_fielding_conditions}
    """
    c.execute(fielding_query, match_ids + [country_id])
    fielding_data = c.fetchone()

    fielding_labels = [
        'Catch', 'Run Out', 'Drop Catch', 'Boundary Save',
        'Clean Stop/Pick Up', 'Direct Hit', 'Missed Catch', 'Missed Run Out',
        'Fumble', 'Missed Fielding', 'Overthrow', 'Taken Half Chance', 'Missed Half Chance'
    ]

    total_ir = 0
    for label, count in zip(fielding_labels, fielding_data):
        stats['fielding'][label] = count or 0
        total_ir += (count or 0) * fielding_weights.get(label, 0)

    # Total Balls Fielded
    balls_fielded_query = f"""
    SELECT COUNT(DISTINCT bfe.ball_id)
    FROM ball_fielding_events bfe
    JOIN ball_events be ON bfe.ball_id = be.ball_id
    JOIN innings i ON be.innings_id = i.innings_id
    JOIN matches m ON i.match_id = m.match_id
    JOIN players p ON be.fielder_id = p.player_id
    WHERE {global_fielding_conditions}
    """
    c.execute(balls_fielded_query, match_ids + [country_id])
    stats['fielding']['Total Balls Fielded'] = c.fetchone()[0] or 0

    # Expected vs Actual Runs
    expected_actual_query = f"""
    SELECT 
        COALESCE(SUM(be.expected_runs), 0),
        COALESCE(SUM(be.runs + be.byes), 0)
    FROM ball_fielding_events bfe
    JOIN ball_events be ON bfe.ball_id = be.ball_id
    JOIN innings i ON be.innings_id = i.innings_id
    JOIN matches m ON i.match_id = m.match_id
    JOIN players p ON be.fielder_id = p.player_id
    WHERE {global_fielding_conditions}
    """
    c.execute(expected_actual_query, match_ids + [country_id])
    expected_runs, actual_runs = c.fetchone()
    stats['fielding']['Expected Runs'] = expected_runs or 0
    stats['fielding']['Actual Runs'] = actual_runs or 0
    stats['fielding']['Runs Saved/Allowed'] = expected_runs - actual_runs

    # Conversion Rate and Pressure Score
    c_ = stats['fielding']['Catch']
    r_ = stats['fielding']['Run Out']
    d_ = stats['fielding']['Drop Catch']
    b_ = stats['fielding']['Boundary Save']
    cs_ = stats['fielding']['Clean Stop/Pick Up']
    dh_ = stats['fielding']['Direct Hit']
    mc_ = stats['fielding']['Missed Catch']
    mru_ = stats['fielding']['Missed Run Out']
    f_ = stats['fielding']['Fumble']
    mf_ = stats['fielding']['Missed Fielding']
    o_ = stats['fielding']['Overthrow']
    thc_ = stats['fielding']['Taken Half Chance']
    mhc_ = stats['fielding']['Missed Half Chance']

    opportunities = c_ + d_ + mc_ + r_ + mru_ + 0.5 * thc_ + 0.5 * mhc_
    successful = c_ + r_ + thc_
    stats['fielding']['Conversion Rate'] = round(((successful / opportunities) * 100 if opportunities > 0 else 0), 2)
    stats['fielding']['Pressure Score'] = dh_ + cs_ + b_ - o_ - mf_ - f_
    stats['fielding']['Fielding Impact Rating'] = total_ir

    total_balls_fielded = stats['fielding']['Total Balls Fielded']
    clean_pickups = stats['fielding']['Clean Stop/Pick Up']

    stats['fielding']['Clean Hands %'] = round(
        (clean_pickups / total_balls_fielded) * 100 if total_balls_fielded > 0 else 0,
        1
    )



    conn.close()

    print(f"Returning stats for {country}:")
    import pprint
    pprint.pprint(stats)

    return stats

def get_player_stats(player_id, tournaments, selected_stats, selected_phases, bowler_type, bowling_arm, team_category, selected_matches=None):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ✅ Get tournament IDs
    c.execute(f"SELECT tournament_id FROM tournaments WHERE tournament_name IN ({','.join(['?']*len(tournaments))})", tournaments)
    tournament_ids = [row[0] for row in c.fetchall()]
    if not tournament_ids:
        return defaultdict(lambda: defaultdict(float))

    # ✅ Get matches for those tournaments
    c.execute(f"""
        SELECT m.match_id
        FROM matches m
        JOIN countries c1 ON m.team_a = c1.country_id
        JOIN countries c2 ON m.team_b = c2.country_id
        WHERE m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})
    """, tournament_ids)

    match_ids = [row[0] for row in c.fetchall()]
    if not match_ids:
        print(f"❌ No matches found for player {player_id}")
        return defaultdict(lambda: defaultdict(float))

    # Apply frontend match filter here if provided
    if selected_matches:
        filtered_match_ids = [m for m in match_ids if m in selected_matches]
        if not filtered_match_ids:
            print("❌ No matches after applying selectedMatches filter")
            return defaultdict(lambda: defaultdict(float))
        match_ids = filtered_match_ids

    match_filter = f"i.match_id IN ({','.join(['?'] * len(match_ids))})"

    # Phase filter
    phase_conditions = {
        'Powerplay': 'be.is_powerplay = 1',
        'Middle Overs': 'be.is_middle_overs = 1',
        'Death Overs': 'be.is_death_overs = 1'
    }
    phase_clauses = [phase_conditions[p] for p in selected_phases if p in phase_conditions]
    phase_filter = f"({' OR '.join(phase_clauses)})" if phase_clauses else "1=1"

    # Bowler filters
    bowler_type_conditions = {
        "Pace": "p.bowling_style = 'Pace'",
        "Medium": "p.bowling_style = 'Medium'",
        "Leg Spin": "p.bowling_style = 'Leg Spin'",
        "Off Spin": "p.bowling_style = 'Off Spin'"
    }
    bowling_arm_conditions = {
        "Left": "p.bowling_arm = 'Left'",
        "Right": "p.bowling_arm = 'Right'"
    }

    type_clauses = [bowler_type_conditions[bt] for bt in bowler_type if bt in bowler_type_conditions]
    arm_clauses = [bowling_arm_conditions[arm] for arm in bowling_arm if arm in bowling_arm_conditions]
    combined_filter_parts = []
    if type_clauses:
        combined_filter_parts.append("(" + " OR ".join(type_clauses) + ")")
    if arm_clauses:
        combined_filter_parts.append("(" + " OR ".join(arm_clauses) + ")")
    combined_filter = " AND ".join(combined_filter_parts)

    # Global filters
    global_batting_conditions = f"{match_filter} AND be.batter_id = ? AND {phase_filter}"
    global_bowling_conditions = f"{match_filter} AND be.bowler_id = ? AND {phase_filter}"
    global_fielding_conditions = f"{match_filter} AND p.player_id = ? AND {phase_filter}"

    stats = defaultdict(lambda: defaultdict(float))

    ### Batting Query
    batting_query = f"""
        SELECT
            COUNT(DISTINCT be.batter_id || '-' || be.innings_id) AS batting_innings,
            SUM(be.runs) AS runs_off_bat,
            SUM(be.wides) + SUM(be.no_balls) + SUM(be.byes) + SUM(be.leg_byes) AS extras,
            SUM(CASE 
                WHEN be.wides = 0 THEN 1 ELSE 0
            END) AS balls_faced,
            SUM(be.dot_balls) AS dot_balls,
            SUM(CASE WHEN be.runs = 1 THEN 1 ELSE 0 END) AS ones,
            SUM(CASE WHEN be.runs = 2 THEN 1 ELSE 0 END) AS twos,
            SUM(CASE WHEN be.runs = 3 THEN 1 ELSE 0 END) AS threes,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            SUM(CASE 
                WHEN be.dismissed_player_id = ? 
                AND be.dismissal_type IS NOT NULL 
                AND LOWER(be.dismissal_type) != 'not out'
                THEN 1 ELSE 0 
            END) AS dismissals,
            SUM(CASE WHEN LOWER(be.shot_type) = 'attacking' THEN 1 ELSE 0 END) AS attacking,
            SUM(CASE WHEN LOWER(be.shot_type) = 'defensive' THEN 1 ELSE 0 END) AS defensive,
            SUM(CASE WHEN LOWER(be.shot_type) = 'rotation' THEN 1 ELSE 0 END) AS rotation,
            AVG(be.batting_intent_score) AS avg_intent
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE {global_batting_conditions}
    """

    c.execute(batting_query, match_ids + [player_id, player_id])
    batting_data = c.fetchone()
    if batting_data:
        stats['batting']['Innings'] = batting_data[0] or 0
        stats['batting']['Runs Off Bat'] = batting_data[1] or 0
        stats['batting']['Batting Extras'] = batting_data[2] or 0
        stats['batting']['Total Runs'] = stats['batting']['Runs Off Bat'] + stats['batting']['Batting Extras']
        stats['batting']['Balls Faced'] = batting_data[3] or 0
        stats['batting']['Dot Balls Faced'] = batting_data[4] or 0
        stats['batting']['1s'] = batting_data[5] or 0
        stats['batting']['2s'] = batting_data[6] or 0
        stats['batting']['3s'] = batting_data[7] or 0
        stats['batting']['4s'] = batting_data[8] or 0
        stats['batting']['6s'] = batting_data[9] or 0
        stats['batting']['Dismissals'] = batting_data[10] or 0

        if stats['batting']['Balls Faced'] > 0:
            stats['batting']['Strike Rate'] = round(
                (stats['batting']['Runs Off Bat'] * 100 / stats['batting']['Balls Faced']), 2)
            stats['batting']['Scoring Shot %'] = round(
                (1 - (stats['batting']['Dot Balls Faced'] / stats['batting']['Balls Faced'])) * 100, 2)

        if stats['batting']['Dismissals'] > 0:
            stats['batting']['Batters Average'] = round(
                stats['batting']['Runs Off Bat'] / stats['batting']['Dismissals'], 2)

        total_intent = sum(filter(None, [batting_data[11], batting_data[12], batting_data[13]]))
        if total_intent > 0:
            stats['batting']['Attacking Shot %'] = round((batting_data[11] / total_intent) * 100, 2)
            stats['batting']['Defensive Shot %'] = round((batting_data[12] / total_intent) * 100, 2)
            stats['batting']['Rotation Shot %'] = round((batting_data[13] / total_intent) * 100, 2)

        if batting_data[14] is not None:
            stats['batting']['Avg Intent Score'] = round(batting_data[14], 2)


    ### Bowling Query
    bowling_query = f"""
        SELECT
            COUNT(*) AS total_balls,
            SUM(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 ELSE 0 END) AS legal_balls,
            SUM(be.runs) + SUM(be.wides) + SUM(be.no_balls) AS runs_conceded,
            SUM(CASE 
                WHEN be.dismissal_type IS NOT NULL 
                AND LOWER(be.dismissal_type) NOT IN ('run out', 'retired out', 'obstructing the field', 'retired not out') 
                AND be.dismissed_player_id = be.batter_id
                THEN 1 ELSE 0 
            END) AS wickets,
            SUM(CASE 
                WHEN be.runs = 0 AND be.wides = 0 AND be.no_balls = 0 
                THEN 1 ELSE 0 
            END) AS dot_balls,
            SUM(be.wides + be.no_balls) AS extras,
            SUM(CASE WHEN be.runs IN (4,6) THEN 1 ELSE 0 END) AS boundaries
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players p ON be.bowler_id = p.player_id
        WHERE {global_bowling_conditions} {f' AND {combined_filter}' if combined_filter else ''}
    """
    c.execute(bowling_query, match_ids + [player_id])
    bowling_data = c.fetchone()

    if bowling_data:
        total_balls = bowling_data[0] or 0
        legal_balls = bowling_data[1] or 0
        stats['bowling']['Overs'] = f"{legal_balls // 6}.{legal_balls % 6}"
        stats['bowling']['Runs Conceded'] = bowling_data[2]
        stats['bowling']['Wickets'] = bowling_data[3]
        stats['bowling']['Dot Balls Bowled'] = bowling_data[4]
        stats['bowling']['Extras'] = bowling_data[5]
        stats['bowling']['Boundaries Conceded'] = bowling_data[6]

        if legal_balls > 0:
            stats['bowling']['Economy'] = round(bowling_data[2] / (legal_balls / 6), 2)
            stats['bowling']['Dot Ball %'] = round((bowling_data[4] / total_balls) * 100, 2)
            if bowling_data[3] > 0:
                stats['bowling']['Bowlers Average'] = round(bowling_data[2] / bowling_data[3], 2)

    ### Fielding Query
    fielding_weights = {
        'Taken Half Chance': 5,
        'Catch': 3,
        'Run Out': 3,
        'Direct Hit': 2,
        'Clean Stop/Pick Up': 1,
        'Boundary Save': 2,
        'Drop Catch': -3,
        'Missed Run Out': -2,
        'Missed Catch': -2,
        'Missed Fielding': -1,
        'Missed Half Chance': -0.5,
        'Fumble': -1,
        'Overthrow': -2
    }

    fielding_query = f"""
        SELECT
            SUM(CASE WHEN fe.event_name = 'Catch' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Run Out' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Drop Catch' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Boundary Save' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Clean Stop/Pick Up' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Direct Hit' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Missed Catch' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Missed Run Out' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Fumble' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Missed Fielding' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Overthrow' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Taken Half Chance' THEN 1 ELSE 0 END),
            SUM(CASE WHEN fe.event_name = 'Missed Half Chance' THEN 1 ELSE 0 END)
        FROM ball_fielding_events bfe
        JOIN fielding_events fe ON bfe.event_id = fe.event_id
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players p ON be.fielder_id = p.player_id
        WHERE {global_fielding_conditions}
    """
    c.execute(fielding_query, match_ids + [player_id])
    fielding_data = c.fetchone()

    fielding_labels = [
        'Catch', 'Run Out', 'Drop Catch', 'Boundary Save',
        'Clean Stop/Pick Up', 'Direct Hit', 'Missed Catch', 'Missed Run Out',
        'Fumble', 'Missed Fielding', 'Overthrow', 'Taken Half Chance', 'Missed Half Chance'
    ]

    total_ir = 0
    for label, count in zip(fielding_labels, fielding_data):
        stats['fielding'][label] = count or 0
        total_ir += (count or 0) * fielding_weights.get(label, 0)

    # Total Balls Fielded
    balls_fielded_query = f"""
        SELECT COUNT(DISTINCT bfe.ball_id)
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players p ON be.fielder_id = p.player_id
        WHERE {global_fielding_conditions}
    """
    c.execute(balls_fielded_query, match_ids + [player_id])
    stats['fielding']['Total Balls Fielded'] = c.fetchone()[0] or 0

    # Expected vs Actual Runs
    expected_actual_query = f"""
        SELECT 
            COALESCE(SUM(be.expected_runs), 0),
            COALESCE(SUM(be.runs + be.byes), 0)
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players p ON be.fielder_id = p.player_id
        WHERE {global_fielding_conditions}
    """
    c.execute(expected_actual_query, match_ids + [player_id])
    expected_runs, actual_runs = c.fetchone()
    stats['fielding']['Expected Runs'] = expected_runs or 0
    stats['fielding']['Actual Runs'] = actual_runs or 0
    stats['fielding']['Runs Saved/Allowed'] = expected_runs - actual_runs

    # Conversion Rate and Pressure Score
    c_ = stats['fielding']['Catch']
    r_ = stats['fielding']['Run Out']
    d_ = stats['fielding']['Drop Catch']
    b_ = stats['fielding']['Boundary Save']
    cs_ = stats['fielding']['Clean Stop/Pick Up']
    dh_ = stats['fielding']['Direct Hit']
    mc_ = stats['fielding']['Missed Catch']
    mru_ = stats['fielding']['Missed Run Out']
    f_ = stats['fielding']['Fumble']
    mf_ = stats['fielding']['Missed Fielding']
    o_ = stats['fielding']['Overthrow']
    thc_ = stats['fielding']['Taken Half Chance']
    mhc_ = stats['fielding']['Missed Half Chance']

    opportunities = c_ + d_ + mc_ + r_ + mru_ + 0.5 * thc_ + 0.5 * mhc_
    successful = c_ + r_ + thc_
    stats['fielding']['Conversion Rate'] = round(((successful / opportunities) * 100 if opportunities > 0 else 0), 2)
    stats['fielding']['Pressure Score'] = dh_ + cs_ + b_ - o_ - mf_ - f_
    stats['fielding']['Fielding Impact Rating'] = total_ir

    total_balls_fielded = stats['fielding']['Total Balls Fielded']
    clean_pickups = stats['fielding']['Clean Stop/Pick Up']

    stats['fielding']['Clean Hands %'] = round(
        (clean_pickups / total_balls_fielded) * 100 if total_balls_fielded > 0 else 0,
        1
    )

    conn.close()

    return stats

def fetch_over_pressure(conn, team_names, match_ids, selected_phases):
    print("✅ fetch_over_pressure called with:", team_names, match_ids, selected_phases)
    
    batting_pressure = defaultdict(lambda: defaultdict(list))
    bowling_pressure = defaultdict(lambda: defaultdict(list))

    phase_filter = ""
    if selected_phases:
        phase_conditions = {
            'Powerplay': 'be.is_powerplay = 1',
            'Middle Overs': 'be.is_middle_overs = 1',
            'Death Overs': 'be.is_death_overs = 1'
        }
        phase_clauses = [phase_conditions[p] for p in selected_phases if p in phase_conditions]
        if phase_clauses:
            phase_filter = f"AND ({' OR '.join(phase_clauses)})"

    cursor = conn.cursor()

    # ✅ Batting pressure (using batting_team as country_name directly)
    cursor.execute(f"""
        SELECT 
            i.batting_team,
            CAST(be.over_number AS INT) + 1 AS over,
            AVG(be.batting_bpi)
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
          AND i.batting_team IN ({','.join(['?'] * len(team_names))})
          {phase_filter}
        GROUP BY i.batting_team, over
    """, match_ids + team_names)
    for team_name, over, avg in cursor.fetchall():
        batting_pressure[team_name][over] = round(avg, 2)

    # ✅ Bowling pressure (using bowling_team as country_name directly)
    cursor.execute(f"""
        SELECT 
            i.bowling_team,
            CAST(be.over_number AS INT) + 1 AS over,
            AVG(be.bowling_bpi)
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
          AND i.bowling_team IN ({','.join(['?'] * len(team_names))})
          {phase_filter}
        GROUP BY i.bowling_team, over
    """, match_ids + team_names)
    for team_name, over, avg in cursor.fetchall():
        bowling_pressure[team_name][over] = round(avg, 2)

    # ✅ Pad each team with over 1 to 20 (None where no value)
    def pad_pressure(raw):
        padded = {}
        for team, over_map in raw.items():
            over_list = [over_map.get(i, None) for i in range(1, 21)]
            padded[team] = over_list
        return padded

    final_batting = pad_pressure(batting_pressure)
    final_bowling = pad_pressure(bowling_pressure)

    return final_batting, final_bowling

def fetch_phase_pressure(conn, team_names, match_ids, selected_phases):
    print("✅ fetch_phase_pressure called with:", team_names, match_ids, selected_phases)
    cursor = conn.cursor()

    phase_pressure_result = {
        "batting": [],
        "bowling": []
    }

    phases = ["Powerplay", "Middle Overs", "Death Overs"]
    phase_column_map = {
        "Powerplay": "is_powerplay",
        "Middle Overs": "is_middle_overs",
        "Death Overs": "is_death_overs"
    }

    for team_name in team_names:
        batting_values = []
        bowling_values = []

        for phase in phases:
            col = phase_column_map[phase]

            # ✅ First: get batting pressure when team is batting
            cursor.execute(f"""
                SELECT COALESCE(SUM(be.batting_bpi), 0)
                FROM ball_events be
                JOIN innings i ON be.innings_id = i.innings_id
                JOIN matches m ON i.match_id = m.match_id
                WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
                AND i.batting_team = ?
                AND be.{col} = 1
            """, match_ids + [team_name])
            batting_bpi = cursor.fetchone()[0]

            # ✅ Second: get bowling pressure when team is bowling
            cursor.execute(f"""
                SELECT COALESCE(SUM(be.bowling_bpi), 0)
                FROM ball_events be
                JOIN innings i ON be.innings_id = i.innings_id
                JOIN matches m ON i.match_id = m.match_id
                WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
                AND i.bowling_team = ?
                AND be.{col} = 1
            """, match_ids + [team_name])
            bowling_bpi = cursor.fetchone()[0]

            batting_values.append(round(batting_bpi, 2))
            bowling_values.append(round(bowling_bpi, 2))

            print(f"🌀 Team {team_name}, Phase {phase}: Batting = {batting_bpi}, Bowling = {bowling_bpi}")

        phase_pressure_result["batting"].append({
            "team": team_name,
            "values": batting_values
        })
        phase_pressure_result["bowling"].append({
            "team": team_name,
            "values": bowling_values
        })

    return phase_pressure_result

def fetch_top_bottom_players(conn, match_ids, team_names):
    result = {
        "batting": {"top": [], "bottom": []},
        "bowling": {"top": [], "bottom": []},
        "fielding": {"top": [], "bottom": []},
        "total": {"top": [], "bottom": []}
    }

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT 
            p.player_name,
            p.role,
            p.player_id,
            p.country_id,
            c.country_name,
            team_role,
            SUM(CASE WHEN pressure_type = 'pressure_applied' THEN pressure_value ELSE 0 END) AS applied,
            SUM(CASE WHEN pressure_type = 'pressure_relieved' THEN ABS(pressure_value) ELSE 0 END) AS relieved
        FROM player_pressure_impact ppi
        JOIN players p ON ppi.player_id = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        JOIN ball_events be ON be.ball_id = ppi.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
        AND c.country_name IN ({','.join(['?'] * len(team_names))})
        GROUP BY p.player_name, ppi.team_role, ppi.player_id
    """, match_ids + team_names)


    players = cursor.fetchall()

    # Separate lists per role + total_map for aggregation
    impact_by_role = defaultdict(list)
    total_impact_map = defaultdict(lambda: {
        "player_name": "",
        "net_impact": 0.0
    })

    for name, role, player_id, country_ids, country_name, team_role, applied, relieved in players:

        net_impact = relieved - applied

            # Flip sign for bowling players only
        if team_role == "bowling":
            net_impact = -net_impact

        # Flip sign for bowling players only
        if team_role == "fielding":
            net_impact = -net_impact

        # Role-specific
        impact_by_role[team_role].append({
            "player_name": name,
            "country": country_name,
            "net_impact": round(net_impact, 2)
        })

        # Aggregated total
        total_impact_map[player_id]["player_name"] = name
        total_impact_map[player_id]["country"] = country_name
        total_impact_map[player_id]["net_impact"] += round(net_impact, 2)

    # Build top/bottom 3 for each role
    for role in ["batting", "bowling", "fielding"]:
        sorted_players = sorted(impact_by_role[role], key=lambda x: x["net_impact"], reverse=True)
        result[role]["top"] = sorted_players[:3]

        sorted_bottom = sorted(impact_by_role[role], key=lambda x: x["net_impact"])
        result[role]["bottom"] = sorted_bottom[:3]

    # Now calculate total impact top/bottom 3
    total_list = list(total_impact_map.values())
    sorted_total = sorted(total_list, key=lambda x: x["net_impact"], reverse=True)
    result["total"]["top"] = sorted_total[:3]
    result["total"]["bottom"] = sorted(total_list, key=lambda x: x["net_impact"])[:3]

    return result

def get_pressure_analysis(payload: PressurePayload):
    print("\U0001F680 Running get_pressure_analysis")
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    team_names = [payload.country1, payload.country2]

    # ✅ Get tournament IDs
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row[0] for row in cursor.fetchall()]

    if not tournament_ids:
        conn.close()
        return {"error": "No matching tournaments found."}

    # ✅ Get match IDs with teamCategory and team name filtering
    if payload.allMatchesSelected:
        team_category_likes = [f"%{payload.teamCategory}%", f"{payload.teamCategory}%"]

        cursor.execute(f"""
            SELECT m.match_id
            FROM matches m
            JOIN countries c1 ON m.team_a = c1.country_id
            JOIN countries c2 ON m.team_b = c2.country_id
            WHERE m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})
            AND (
                c1.country_name LIKE ? OR c1.country_name LIKE ?
                OR c2.country_name LIKE ? OR c2.country_name LIKE ?
            )
            AND (
                c1.country_name IN ({','.join(['?'] * len(team_names))})
                OR c2.country_name IN ({','.join(['?'] * len(team_names))})
            )
        """, (
            tournament_ids +
            team_category_likes * 2 +  # Applies to both c1 and c2
            team_names * 2             # Applies to both c1 and c2
        ))

        match_ids = [row[0] for row in cursor.fetchall()]
    else:
        match_ids = payload.selectedMatches

    if not match_ids:
        print("❌ No matching matches found — check tournament/team name/category filters")
        conn.close()
        return {"error": "No matching matches found."}


    # ✅ Fetch pressure data using country names
    batting_pressure, bowling_pressure = fetch_over_pressure(conn, team_names, match_ids, payload.selectedPhases)
    phase_pressure = fetch_phase_pressure(conn, team_names, match_ids, payload.selectedPhases)
    top_bottom_players = fetch_top_bottom_players(conn, match_ids, team_names)

    conn.close()
    print("✅ Finished calculating pressure data")
    return {
        "overPressure": {
            "batting": batting_pressure,
            "bowling": bowling_pressure
        },
        "phasePressure": phase_pressure,
        "topBottomPlayers": top_bottom_players
    }

def get_wagon_wheel_data(payload: WagonWheelPayload):
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ✅ Get team IDs
    team_map = {}
    for name in [payload.country1, payload.country2]:
        cursor.execute("SELECT country_id FROM countries WHERE country_name = ?", (name,))
        row = cursor.fetchone()
        if row:
            team_map[name] = row[0]

    if len(team_map) != 2:
        conn.close()
        return {"error": "Could not resolve both countries."}

    # ✅ Get tournament IDs
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row[0] for row in cursor.fetchall()]

    if not tournament_ids:
        conn.close()
        return {"error": "No tournaments found."}

    # ✅ Get match IDs with enhanced teamCategory filtering
    if payload.allMatchesSelected:
        country_like_1 = f"%{payload.teamCategory}%"  # Handles names like "Training 1 Women"
        country_like_2 = f"{payload.teamCategory}%"   # Handles names like "TrainingWomen" or "Training - Squad A"

        country_filters = [country_like_1, country_like_2, country_like_1, country_like_2]

        cursor.execute(f"""
            SELECT m.match_id
            FROM matches m
            JOIN countries c1 ON m.team_a = c1.country_id
            JOIN countries c2 ON m.team_b = c2.country_id
            WHERE m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})
            AND (
                c1.country_name LIKE ? OR c1.country_name LIKE ? OR
                c2.country_name LIKE ? OR c2.country_name LIKE ?
            )
            AND (m.team_a IN (?, ?) OR m.team_b IN (?, ?))
        """, tournament_ids + country_filters + list(team_map.values()) * 2)

        match_ids = [row[0] for row in cursor.fetchall()]
    else:
        match_ids = payload.selectedMatches

    if not match_ids:
        print("❌ No matches found — likely teamCategory mismatch or tournament mismatch")
        conn.close()
        return {"error": "No matches found."}


    # ✅ Phase filter
    phase_map = {
        "Powerplay": "be.is_powerplay = 1",
        "Middle Overs": "be.is_middle_overs = 1",
        "Death Overs": "be.is_death_overs = 1"
    }
    phase_clauses = [phase_map[p] for p in payload.selectedPhases if p in phase_map]
    phase_filter = f"AND ({' OR '.join(phase_clauses)})" if phase_clauses else ""

    result = {}

    for team_name, team_id in team_map.items():
        query = f"""
            SELECT 
                be.shot_x,
                be.shot_y,
                be.runs
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players bp ON be.batter_id = bp.player_id
            JOIN players bw ON be.bowler_id = bw.player_id
            WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
              AND bp.country_id = ?
              AND be.shot_x IS NOT NULL
              AND be.shot_y IS NOT NULL
              {phase_filter}
        """

        params = match_ids + [team_id]

        # ✅ Batting filters (applied to batter)
        if payload.selectedBattingHands:
            query += f" AND bp.batting_hand IN ({','.join(['?'] * len(payload.selectedBattingHands))})"
            params.extend(payload.selectedBattingHands)

        # ✅ Bowling arm filter (applied to bowler)
        if payload.selectedBowlingArms:
            query += f" AND bw.bowling_arm IN ({','.join(['?'] * len(payload.selectedBowlingArms))})"
            params.extend(payload.selectedBowlingArms)

        # ✅ Bowling style filter (applied to bowler)
        if payload.selectedBowlerTypes:
            query += f" AND bw.bowling_style IN ({','.join(['?'] * len(payload.selectedBowlerTypes))})"
            params.extend(payload.selectedBowlerTypes)

        # Define the full set of lengths
        all_lengths = {"Full", "Good", "Short", "Full Toss", "Yorker"}

        # 🚫 Case 1: If user deselects all length filters, return nothing
        if not payload.selectedLengths:
            return {team: [] for team in team_map.keys()}  # or return empty dict

        # ✅ Case 2: If user selects all lengths, skip filtering
        elif set(payload.selectedLengths) == all_lengths:
            pass  # No length filter needed

        # 🎯 Case 3: Filter selected lengths (1–4 selected)
        else:
            spin_map = {
                "Full Toss": (0.0, 0.1),
                "Yorker": (0.1, 0.25),
                "Full": (0.25, 0.4),
                "Good": (0.4, 0.6),
                "Short": (0.6, 1.0)
            }
            pace_map = {
                "Full Toss": (0.0, 0.1),
                "Yorker": (0.1, 0.25),
                "Full": (0.25, 0.4),
                "Good": (0.4, 0.6),
                "Short": (0.6, 1.0)
            }

            length_clauses = []
            length_params = []

            for length in payload.selectedLengths:
                if length in spin_map:
                    min_s, max_s = spin_map[length]
                    length_clauses.append("(bw.bowling_style = 'Spin' AND be.pitch_y BETWEEN ? AND ?)")
                    length_params.extend([min_s, max_s])
                if length in pace_map:
                    min_p, max_p = pace_map[length]
                    length_clauses.append("(bw.bowling_style IN ('Pace', 'Medium') AND be.pitch_y BETWEEN ? AND ?)")
                    length_params.extend([min_p, max_p])

            if length_clauses:
                query += f" AND ({' OR '.join(length_clauses)})"
                params.extend(length_params)



        cursor.execute(query, params)
        shots = cursor.fetchall()
        result[team_name] = [{"x": x, "y": y, "runs": r if r is not None else 0} for x, y, r in shots]

    conn.close()
    return result

def get_pitch_map_data(payload: PitchMapPayload):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ✅ Resolve team IDs
    team_map = {}
    for country in [payload.country1, payload.country2]:
        cursor.execute("SELECT country_id FROM countries WHERE country_name = ?", (country,))
        row = cursor.fetchone()
        if row:
            team_map[country] = row[0]
    if len(team_map) != 2:
        conn.close()
        return {"error": "Could not resolve both countries."}

    # ✅ Resolve tournament IDs
    cursor.execute(f"""
        SELECT tournament_id FROM tournaments
        WHERE tournament_name IN ({','.join(['?'] * len(payload.tournaments))})
    """, payload.tournaments)
    tournament_ids = [row[0] for row in cursor.fetchall()]
    if not tournament_ids:
        conn.close()
        return {"error": "No tournaments found."}

    # ✅ Resolve match IDs
    if payload.allMatchesSelected:
        team_category_likes = [f"%{payload.teamCategory}%", f"{payload.teamCategory}%"]

        cursor.execute(f"""
            SELECT m.match_id
            FROM matches m
            JOIN countries c1 ON m.team_a = c1.country_id
            JOIN countries c2 ON m.team_b = c2.country_id
            WHERE m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})
            AND (
                c1.country_name LIKE ? OR c1.country_name LIKE ?
                OR c2.country_name LIKE ? OR c2.country_name LIKE ?
            )
            AND (
                m.team_a IN (?, ?) OR m.team_b IN (?, ?)
            )
        """, tournament_ids + team_category_likes * 2 + list(team_map.values()) * 2)

        match_ids = [row[0] for row in cursor.fetchall()]
    else:
        match_ids = payload.selectedMatches

    if not match_ids:
        print(f"❌ No matches found for countries: {list(team_map.keys())}, category: {payload.teamCategory}, tournaments: {payload.tournaments}")
        conn.close()
        return {"error": "No matches found."}


    # ✅ Phase filter
    phase_map = {
        "Powerplay": "be.is_powerplay = 1",
        "Middle Overs": "be.is_middle_overs = 1",
        "Death Overs": "be.is_death_overs = 1"
    }
    phase_clauses = [phase_map[p] for p in payload.selectedPhases if p in phase_map]
    phase_filter = f"AND ({' OR '.join(phase_clauses)})" if phase_clauses else ""

    result = {}

    for team_name in team_map:
        query = f"""
            SELECT 
                be.pitch_x,
                be.pitch_y,
                be.runs,
                be.wides,
                be.no_balls,
                be.dismissal_type
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players bw ON be.bowler_id = bw.player_id
            JOIN players bp ON be.batter_id = bp.player_id
            WHERE m.match_id IN ({','.join(['?'] * len(match_ids))})
              AND i.bowling_team = ?
              AND be.pitch_x IS NOT NULL
              AND be.pitch_y IS NOT NULL
              {phase_filter}
        """

        params = match_ids + [team_name]
        print(f"🔍 Pitch Map Query for {team_name}")
        print("Match IDs:", match_ids)
        print("Params:", params)

        # ✅ Bowling Arm filter
        if payload.selectedBowlingArms:
            query += f" AND bw.bowling_arm IN ({','.join(['?'] * len(payload.selectedBowlingArms))})"
            params.extend(payload.selectedBowlingArms)

        # ✅ Bowling Style filter
        if payload.selectedBowlerTypes:
            query += f" AND bw.bowling_style IN ({','.join(['?'] * len(payload.selectedBowlerTypes))})"
            params.extend(payload.selectedBowlerTypes)

        # ✅ Batting Hand filter
        if payload.selectedBattingHands:
            query += f" AND bp.batting_hand IN ({','.join(['?'] * len(payload.selectedBattingHands))})"
            params.extend(payload.selectedBattingHands)

        cursor.execute(query, params)
        balls = cursor.fetchall()
        print(f"🎯 Pitch Map Rows Returned for {team_name}: {len(balls)}")
        for b in balls[:5]:
            print("   Example Ball:", b)
        result[team_name] = [
            {
                "pitch_x": x,
                "pitch_y": y,
                "runs": r if r is not None else 0,
                "wides": w if w is not None else 0,
                "no_balls": nb if nb is not None else 0,
                "dismissal_type": d
            }
            for x, y, r, w, nb, d in balls
        ]


    conn.close()
    return result

def get_individual_wagon_wheel_data(player_id, batting_team, tournament_ids):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    tournament_filter = ""
    if tournament_ids:
        tournament_filter = f" AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    query = f"""
        SELECT 
            be.shot_x AS x, 
            be.shot_y AS y, 
            be.runs, 
            be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.batter_id = ?
        AND i.batting_team = ?
        {tournament_filter}
        AND be.shot_x IS NOT NULL AND be.shot_y IS NOT NULL
    """

    params = [player_id, batting_team] + tournament_ids
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]

def get_individual_pitch_map_data(player_id, bowling_team, tournament_ids):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    tournament_filter = ""
    if tournament_ids:
        tournament_filter = f" AND m.tournament_id IN ({','.join(['?'] * len(tournament_ids))})"

    query = f"""
        SELECT 
            be.pitch_x AS pitch_x,
            be.pitch_y AS pitch_y,
            be.runs,
            be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN matches m ON i.match_id = m.match_id
        WHERE be.bowler_id = ?
        AND i.bowling_team = ?
        {tournament_filter}
        AND be.pitch_x IS NOT NULL AND be.pitch_y IS NOT NULL
    """

    params = [player_id, bowling_team] + tournament_ids
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]

def fetch_player_match_stats(match_id: int, player_id: int):
    db_path = os.path.join(os.path.dirname(__file__), "cricket_analysis.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Player name
    cursor.execute("SELECT player_name FROM players WHERE player_id = ?", (player_id,))
    player_row = cursor.fetchone()
    if not player_row:
        return None
    player_name = player_row["player_name"]

    # Match info
    cursor.execute("""
        SELECT m.match_date, c1.country_name AS team_a, c2.country_name AS team_b, t.tournament_name
        FROM matches m
        JOIN countries c1 ON m.team_a = c1.country_id
        JOIN countries c2 ON m.team_b = c2.country_id
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE m.match_id = ?
    """, (match_id,))
    match_row = cursor.fetchone()

    # Match result
    cursor.execute("SELECT result FROM matches WHERE match_id = ?", (match_id,))
    match_result_row = cursor.fetchone()
    match_result = match_result_row["result"] if match_result_row else "N/A"

    def convert_partial_overs_to_cricket(overs_float):
        overs_whole = int(overs_float)
        fraction = overs_float - overs_whole
        balls = int(round(fraction * 6))
        return f"{overs_whole}.{balls}"

    # First innings summary
    cursor.execute("""
        SELECT batting_team, total_runs, wickets, overs_bowled
        FROM innings
        WHERE match_id = ? AND innings = 1
    """, (match_id,))
    first_innings_row = cursor.fetchone()
    first_innings_summary = dict(first_innings_row or {})

    # Convert overs_bowled to cricket-style overs (if exists)
    if first_innings_summary.get("overs_bowled") is not None:
        first_innings_summary["overs"] = convert_partial_overs_to_cricket(first_innings_summary["overs_bowled"])
    else:
        first_innings_summary["overs"] = "0.0"

    # Second innings summary
    cursor.execute("""
        SELECT batting_team, total_runs, wickets, overs_bowled
        FROM innings
        WHERE match_id = ? AND innings = 2
    """, (match_id,))
    second_innings_row = cursor.fetchone()
    second_innings_summary = dict(second_innings_row or {})

    # Convert overs_bowled to cricket-style overs (if exists)
    if second_innings_summary.get("overs_bowled") is not None:
        second_innings_summary["overs"] = convert_partial_overs_to_cricket(second_innings_summary["overs_bowled"])
    else:
        second_innings_summary["overs"] = "0.0"




    # Batting summary
    cursor.execute("""
        SELECT
            SUM(be.runs) AS runs,
            COUNT(CASE WHEN be.wides = 0 THEN 1 END) AS balls_faced,  -- excludes wides, includes no balls
            SUM(CASE WHEN be.dismissal_type IS NOT NULL AND LOWER(be.dismissal_type) != 'not out' THEN 1 ELSE 0 END) AS dismissals,
            ROUND(AVG(be.batting_intent_score), 2) AS average_intent,
            (
                SELECT be2.dismissal_type
                FROM ball_events be2
                JOIN innings i2 ON be2.innings_id = i2.innings_id
                WHERE i2.match_id = ? AND be2.dismissed_player_id = ?
                AND be2.dismissal_type IS NOT NULL
                ORDER BY be2.over_number DESC, be2.ball_number DESC
                LIMIT 1
            ) AS dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ?
    """, (match_id, player_id, match_id, player_id))

    batting = dict(cursor.fetchone())

    # Strike Rate & Scoring Shot %
    if batting['balls_faced']:
        batting['strike_rate'] = round(batting['runs'] * 100.0 / batting['balls_faced'], 2)

        # Count scoring shots (legal + no balls only)
        cursor.execute("""
            SELECT COUNT(*)
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND be.batter_id = ? AND be.runs > 0 AND be.wides = 0
        """, (match_id, player_id))
        scoring_shots = cursor.fetchone()[0]
        batting['scoring_shot_percentage'] = round(scoring_shots * 100.0 / batting['balls_faced'], 2)
    else:
        batting.update({"strike_rate": 0, "scoring_shot_percentage": 0})

    # If no dismissal_type found, mark as 'Not out'
    if not batting['dismissal_type']:
        batting['dismissal_type'] = "Not out"


    # Bowling summary
    cursor.execute("""
        SELECT
            COUNT(*) AS total_balls,  -- includes wides and no balls
            COUNT(CASE WHEN be.wides = 0 AND be.no_balls = 0 THEN 1 END) AS legal_balls,
            SUM(CASE
              WHEN be.wides = 0
               AND be.no_balls = 0
               AND COALESCE(be.runs,0) = 0
               AND COALESCE(be.byes,0) = 0
               AND COALESCE(be.leg_byes,0) = 0
               AND COALESCE(be.penalty_runs,0) = 0
                THEN 1 ELSE 0
            END) AS dot_balls,
            SUM(be.runs + be.wides + be.no_balls) AS runs_conceded,
            SUM(
                CASE
                    WHEN be.dismissed_player_id = be.batter_id
                    AND LOWER(be.dismissal_type) NOT IN ('not out', 'run out', 'obstructing the field', 'retired out', 'retired hurt')
                    THEN 1 ELSE 0
                END
            ) AS wickets,
            SUM(be.wides + be.no_balls) AS extras
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.bowler_id = ?
    """, (match_id, player_id))

    bowling = dict(cursor.fetchone())

    if bowling['total_balls']:
        bowling['overs'] = round(bowling['legal_balls'] / 6, 1)
        bowling['dot_ball_percentage'] = round(bowling['dot_balls'] * 100.0 / bowling['total_balls'], 2)
        bowling['economy'] = round(bowling['runs_conceded'] * 6.0 / bowling['legal_balls'], 2)
    else:
        bowling.update({"overs": 0, "dot_ball_percentage": 0, "economy": 0})


    # Fielding summary
    cursor.execute("""
        SELECT
            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id = 1) AS clean_pickups,

            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id = 2) AS catches,

            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id = 3) AS run_outs,

            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ?) AS total_fielding_events
    """, (match_id, player_id, match_id, player_id, match_id, player_id, match_id, player_id))

    fielding_row = cursor.fetchone()

    # Runs saved / allowed
    cursor.execute("""
        SELECT 
            COALESCE(SUM(be.expected_runs), 0) AS expected_runs,
            COALESCE(SUM(
                COALESCE(be.runs,0)
            + CASE WHEN COALESCE(be.wides,0)    > 0 THEN COALESCE(be.wides,0)    - 1 ELSE 0 END
            + CASE WHEN COALESCE(be.no_balls,0) > 0 THEN COALESCE(be.no_balls,0) - 1 ELSE 0 END
            + COALESCE(be.byes,0)
            + COALESCE(be.leg_byes,0)
            + COALESCE(be.penalty_runs,0)
            ), 0) AS adjusted_actual_runs
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ?
        AND EXISTS (
            SELECT 1 FROM fielding_contributions fc 
            WHERE fc.ball_id = be.ball_id AND fc.fielder_id = ?
        )
    """, (match_id, player_id))

    expected_runs, adjusted_actual_runs = cursor.fetchone()
    runs_saved_allowed = (expected_runs or 0) - (adjusted_actual_runs or 0)

    if fielding_row["total_fielding_events"]:
        clean_pickup_pct = round(fielding_row["clean_pickups"] * 100.0 / fielding_row["total_fielding_events"], 2)
    else:
        clean_pickup_pct = 0.0


    cursor.execute("""
        SELECT
            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id = 2) AS catches,
            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id = 3) AS run_outs,
            (SELECT COUNT(*) FROM ball_fielding_events bfe
            JOIN fielding_contributions fc ON bfe.ball_id = fc.ball_id
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id = ? AND fc.fielder_id = ? AND bfe.event_id IN (4, 5, 6)) AS missed_chances
    """, (match_id, player_id, match_id, player_id, match_id, player_id))
    row = cursor.fetchone()
    catches = row["catches"] or 0
    run_outs = row["run_outs"] or 0
    missed_chances = row["missed_chances"] or 0


    chances_taken = catches + run_outs
    total_chances = chances_taken + missed_chances

    if total_chances:
        conversion_percentage = round((chances_taken * 100.0) / total_chances, 2)
        conversion_rate_display = f"{chances_taken}/{total_chances} ({conversion_percentage}%)"
    else:
        conversion_rate_display = "0/0 (0%)"




    # Now extract values using indexing
    fielding = {
        "clean_pickups": fielding_row["clean_pickups"] if fielding_row["clean_pickups"] is not None else 0,
        "catches": fielding_row["catches"] if fielding_row["catches"] is not None else 0,
        "run_outs": fielding_row["run_outs"] if fielding_row["run_outs"] is not None else 0,
        "total_fielding_events": fielding_row["total_fielding_events"] if fielding_row["total_fielding_events"] is not None else 0,
        "runs_saved_allowed": runs_saved_allowed,
        "clean_pickup_percentage": clean_pickup_pct,
        "conversion_rate": conversion_rate_display
    }





    # Ball by ball batting breakdown
    cursor.execute("""
        SELECT be.over_number, be.ball_number, be.runs, be.footwork, be.shot_selection, be.shot_type,
            be.aerial, be.edged, be.ball_missed
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ?
        ORDER BY be.over_number, be.ball_number
    """, (match_id, player_id))
    ball_by_ball_batting = [dict(row) for row in cursor.fetchall()]

    # Scoring shot breakdown
    cursor.execute("""
        SELECT be.runs, COUNT(*) AS count
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id  
        WHERE i.match_id = ? AND be.batter_id = ?
            AND be.wides = 0  -- ✅ only legal deliveries
        GROUP BY be.runs

    """, (match_id, player_id))
    scoring_shots_breakdown = {str(row["runs"]): row["count"] for row in cursor.fetchall()}

    # Off side and leg side run distribution
    cursor.execute("""
        SELECT
            SUM(CASE WHEN be.shot_x < 0 THEN be.runs ELSE 0 END) AS off_side_runs,
            SUM(CASE WHEN be.shot_x >= 0 THEN be.runs ELSE 0 END) AS leg_side_runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ?
    """, (match_id, player_id))

    side_data = cursor.fetchone()
    off_side_runs = side_data['off_side_runs'] or 0
    leg_side_runs = side_data['leg_side_runs'] or 0
    total_runs = off_side_runs + leg_side_runs

    # Calculate percentages
    if total_runs > 0:
        off_side_percentage = round(off_side_runs * 100.0 / total_runs, 2)
        leg_side_percentage = round(leg_side_runs * 100.0 / total_runs, 2)
    else:
        off_side_percentage = leg_side_percentage = 0

    # Fetch raw data to compute ball lengths
    cursor.execute("""
        SELECT be.pitch_y, bw.bowling_style, be.runs, be.wides, be.no_balls,
            be.dismissal_type, be.edged, be.ball_missed, be.shot_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players bw ON be.bowler_id = bw.player_id
        WHERE i.match_id = ? AND be.bowler_id = ?
        ORDER BY be.over_number, be.ball_number
    """, (match_id, player_id))

    zone_maps = {
        "spin": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        },
        "pace": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        }
    }

    ball_by_ball_bowling = []
    for idx, row in enumerate(cursor.fetchall(), start=1):
        pitch_y = row["pitch_y"]
        style = (row["bowling_style"] or "").lower()
        zone_map = zone_maps["spin"] if "spin" in style else zone_maps["pace"]

        length = "Unknown"
        if pitch_y is not None:
            for zone, (start, end) in zone_map.items():
                if start <= pitch_y < end:
                    length = zone
                    break

        total_runs = (row["runs"] or 0) + (row["wides"] or 0) + (row["no_balls"] or 0)

        ball_by_ball_bowling.append({
            "ball_number": idx,
            "runs": row["runs"],
            "extras": (row["wides"] or 0) + (row["no_balls"] or 0),
            "length": length,
            "dismissal_type": row["dismissal_type"] or "-",
            "false_shot": ("Yes" if (row["edged"] or row["ball_missed"]) and row["shot_type"] and row["shot_type"].lower() != "leave" else "No")
        })


    # Compute Zone Effectiveness
    cursor.execute("""
        SELECT
            be.pitch_y,
            bw.bowling_style,
            be.runs,
            be.wides,
            be.no_balls,
            CASE
            WHEN be.wides = 0
            AND be.no_balls = 0
            AND COALESCE(be.runs,0) = 0
            AND COALESCE(be.byes,0) = 0
            AND COALESCE(be.leg_byes,0) = 0
            AND COALESCE(be.penalty_runs,0) = 0
            THEN 1 ELSE 0
            END AS dot_balls,
            be.edged,
            be.ball_missed,
            be.shot_type,
            be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        JOIN players bw ON be.bowler_id = bw.player_id
        WHERE be.bowler_id = ? AND i.match_id = ? AND be.pitch_y IS NOT NULL
    """, (player_id, match_id))

    zone_data = cursor.fetchall()

    zone_labels = ["Full Toss", "Yorker", "Full", "Good", "Short"]
    zone_maps = {
        "spin": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        },
        "pace": {
            "Full Toss": (-0.0909, 0.03636),
            "Yorker": (0.03636, 0.1636),
            "Full": (0.1636, 0.31818),
            "Good": (0.31818, 0.545454),
            "Short": (0.545454, 1.0)
        }
    }

    zone_stats = {
        label: {
            "balls": 0, "runs": 0, "wickets": 0,
            "dots": 0, "false_shots": 0,
            "wides": 0, "no_balls": 0  # new entries!
        }
        for label in zone_labels
    }

    for row in zone_data:
        pitch_y = row["pitch_y"]
        style = (row["bowling_style"] or "").lower()
        zone_map = zone_maps["spin"] if "spin" in style else zone_maps["pace"]

        total_runs = (row["runs"] or 0) + (row["wides"] or 0) + (row["no_balls"] or 0)

        for zone, (start, end) in zone_map.items():
            if start <= pitch_y < end:
                zone_stats[zone]["balls"] += 1
                zone_stats[zone]["runs"] += total_runs
                zone_stats[zone]["dots"] += row["dot_balls"] or 0
                zone_stats[zone]["wides"] += row["wides"] or 0   # new
                zone_stats[zone]["no_balls"] += row["no_balls"] or 0  # new

                if row["dismissal_type"] and row["dismissal_type"].lower() != "not out":
                    zone_stats[zone]["wickets"] += 1
                if (row["edged"] or row["ball_missed"]) and row["shot_type"] and row["shot_type"].lower() != "leave":
                    zone_stats[zone]["false_shots"] += 1
                break

    zone_effectiveness = []
    for zone in zone_labels:
        z = zone_stats[zone]
        balls = z["balls"] or 1  # prevent division by zero
        zone_effectiveness.append({
            "zone": zone,
            "balls": z["balls"],
            "runs": z["runs"],
            "wickets": z["wickets"],
            "avg_runs_per_ball": round(z["runs"] / balls, 2),
            "dot_pct": round((z["dots"] / balls) * 100, 2),
            "false_shot_pct": round((z["false_shots"] / balls) * 100, 2),
            "wides": z["wides"],
            "no_balls": z["no_balls"],  # new
            "dot_balls": z["dots"],     # for direct display
        })


    # Wagon wheel data
    cursor.execute("""
        SELECT be.shot_x, be.shot_y, be.runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND be.batter_id = ? AND be.shot_x IS NOT NULL AND be.shot_y IS NOT NULL
    """, (match_id, player_id))
    wagon_wheel_data = [dict(row) for row in cursor.fetchall()]

    # Pitch map data
    cursor.execute("""
        SELECT 
            be.pitch_x, 
            be.pitch_y,
            be.runs,
            be.wides,
            be.no_balls,
            be.dismissal_type
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? 
        AND be.bowler_id = ? 
        AND be.pitch_x IS NOT NULL 
        AND be.pitch_y IS NOT NULL
    """, (match_id, player_id))
    pitch_map_data = [dict(row) for row in cursor.fetchall()]


    conn.close()

    return {
        "player_name": player_name,
        "match": {**dict(match_row), "result": match_result},
        "first_innings_summary": first_innings_summary,
        "second_innings_summary": second_innings_summary,
        "batting": batting,  # Updated to include off/leg side data and other batting insights
        "bowling": bowling,  # Updated to include dot ball % and economy
        "fielding": fielding,  # Fielding summary (like catches)
        "ball_by_ball_batting": ball_by_ball_batting,  # Full ball-by-ball batting data
        "scoring_shots_breakdown": scoring_shots_breakdown,  # 0s, 1s, 2s breakdown
        "side_runs": {
            "off_side_runs": off_side_runs,
            "leg_side_runs": leg_side_runs,
            "off_side_percentage": off_side_percentage,
            "leg_side_percentage": leg_side_percentage
        },
        "wagon_wheel_data": wagon_wheel_data,  # For your zone-based & line-based rendering
        "ball_by_ball_bowling": ball_by_ball_bowling,  # Full ball-by-ball bowling data
        "pitch_map_data": pitch_map_data,  # Pitch map for the bowler
        "zone_effectiveness": zone_effectiveness  # Zone-wise effectiveness for the bowler
    }

def generate_pdf_report(data: dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    bold = ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10)
    centered = ParagraphStyle(name='Centered', parent=styles['Normal'], fontSize=11, alignment=1)
    elements = []

    # 1️⃣ Header
    elements.append(Paragraph(f"<b>{data['player_name']}</b>", styles['Title']))
    elements.append(Spacer(1, 6))

    # Center-aligned Tournament Name
    elements.append(Paragraph(f"<b>{data['match']['tournament_name']}</b>", centered))
    elements.append(Spacer(1, 4))

    # Team A vs Team B
    elements.append(Paragraph(f"<b>{data['match']['team_a']} vs {data['match']['team_b']}</b>", centered))
    elements.append(Spacer(1, 4))
    # Match Date
    elements.append(Paragraph(f"<b>{data['match']['match_date']}</b>", centered))
    elements.append(Spacer(1, 10))

    # First Innings Score
    first_innings = data.get("first_innings_summary", {})
    first_innings_score = f"{first_innings.get('batting_team', 'N/A')}: {first_innings.get('total_runs', 0)}/{first_innings.get('wickets', 0)} from {first_innings.get('overs', '0')} overs"
    elements.append(Paragraph(first_innings_score, centered))
    elements.append(Spacer(1, 4))

    # Second Innings Score
    second_innings = data.get("second_innings_summary", {})
    second_innings_score = f"{second_innings.get('batting_team', 'N/A')}: {second_innings.get('total_runs', 0)}/{second_innings.get('wickets', 0)} from {second_innings.get('overs', '0')} overs"
    elements.append(Paragraph(second_innings_score, centered))
    elements.append(Spacer(1, 4))

    # Match Result
    elements.append(Paragraph(f"<b>{data['match']['result']}</b>", centered))
    elements.append(Spacer(1, 40))


    # 2️⃣ Batting Summary
    elements.append(Paragraph("<b>Batting Summary</b>", bold))
    elements.append(Spacer(1, 5))
    if data["batting"]["balls_faced"] > 0:
        batting = data['batting']
        if batting:
            batting_table_data = [
                ["Runs", "Balls", "Strike Rate", "Scoring Shot %", "Average Intent", "Dismissal"],
                [
                    batting.get('runs', 0),
                    batting.get('balls_faced', 0),
                    batting.get('strike_rate', 0),
                    batting.get('scoring_shot_percentage', "N/A"),
                    batting.get('average_intent', "N/A"),
                    batting.get('dismissal_type', "Not out")
                ]
            ]
            batting_table = Table(batting_table_data)
            batting_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(batting_table)
        else:
            elements.append(Paragraph("Did not bat", styles['Normal']))
        elements.append(Spacer(1, 30))
    else:
        elements.append(Paragraph("Did not bat", centered))
        elements.append(Spacer(1, 10))

    # 3️⃣ Bowling Summary
    elements.append(Paragraph("<b>Bowling Summary</b>", bold))
    elements.append(Spacer(1, 5))
    if data["bowling"]["total_balls"] > 0:
        bowling = data['bowling']
        if bowling:
            bowling_table_data = [
                ["Overs", "Dot Balls", "Runs Conceded", "Wickets", "Extras", "Dot Ball %", "Economy"],
                [
                    bowling.get('overs', 0),
                    bowling.get('dot_balls', 0),
                    bowling.get('runs_conceded', 0),
                    bowling.get('wickets', 0),
                    bowling.get('extras', 0),
                    bowling.get('dot_ball_percentage', "N/A"),
                    bowling.get('economy', "N/A")
                ]
            ]
            bowling_table = Table(bowling_table_data)
            bowling_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(bowling_table)
        else:
            elements.append(Paragraph("Did not bowl", styles['Normal']))
        elements.append(Spacer(1, 30))
    else:
        elements.append(Paragraph("Did not bowl", centered))
        elements.append(Spacer(1, 10))

    # 4️⃣ Fielding Summary
    elements.append(Paragraph("<b>Fielding Summary</b>", bold))
    elements.append(Spacer(1, 5))
    fielding = data['fielding']
    if fielding:
        fielding_table_data = [
            ["Total Balls Fielded", "Clean Pick Up %", "Catch(es)", "Run Out(s)", "Conversion Rate", "Runs Allowed/Saved"],
            [
                fielding.get('total_fielding_events', 0),
                fielding.get('clean_pickup_percentage', "N/A"),
                fielding.get('catches', 0),
                fielding.get('run_outs', 0),
                fielding.get('conversion_rate', "N/A"),
                fielding.get('runs_saved_allowed', "N/A")
            ]
        ]
        fielding_table = Table(fielding_table_data)
        fielding_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        elements.append(fielding_table)
    else:
        elements.append(Paragraph("No fielding data available", styles['Normal']))
    elements.append(PageBreak())

    # 5️⃣ Detailed Batting Summary
    if data["batting"]["balls_faced"] > 0:
        elements.append(Paragraph("<b>Detailed Batting Summary</b>", styles['Title']))
        elements.append(Spacer(1, 10))

        if os.path.exists("/tmp/wagon_wheel_chart.png"):
            elements.append(Paragraph("<b>Wagon Wheel</b>", bold))
            elements.append(Image("/tmp/wagon_wheel_chart.png", width=300, height=300))
            elements.append(Spacer(1, 6))
            add_wagon_wheel_legend(elements)
            elements.append(Spacer(1, 40))

        # 🟩 Create column layout with labels above each table
        # First column: Scoring Shot Breakdown
        scoring_label = Paragraph("<b>Scoring Shot Breakdown</b>", bold)
        elements.append(Spacer(1, 6))
        score_data = [["Runs", "Count"]] + [[r, c] for r, c in data['scoring_shots_breakdown'].items()]
        scoring_table = Table(score_data)
        scoring_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey)
        ]))

        # Second column: Off/Leg Side Runs
        side_label = Paragraph("<b>Off/Leg Side Run Distribution</b>", bold)
        elements.append(Spacer(1, 6))
        side_data = data.get("side_runs", {})
        side_table_data = [["Side", "Runs", "Percentage"]]
        side_table_data.append([
            "Off Side",
            side_data.get("off_side_runs", 0),
            f"{side_data.get('off_side_percentage', 0)}%"
        ])
        side_table_data.append([
            "Leg Side",
            side_data.get("leg_side_runs", 0),
            f"{side_data.get('leg_side_percentage', 0)}%"
        ])
        side_table = Table(side_table_data)
        side_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey)
        ]))

        # 🟩 Combine into a two-column layout
        combined_table = Table([
            [[scoring_label, scoring_table], [side_label, side_table]]
        ], colWidths=[270, 270])
        combined_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.append(combined_table)
        elements.append(Spacer(1, 10))

        # 🟩 Page break before ball by ball
        elements.append(PageBreak())

        # 🟩 Ball by ball breakdown on next page
        bb_data = [["Ball", "Runs", "Shot", "Footwork", "Shot Type", "Aerial", "Edged", "Missed"]]
        for idx, ball in enumerate(data['ball_by_ball_batting'], start=1):
            bb_data.append([
                idx,
                ball.get("runs", "N/A"),
                ball.get("shot_selection", "N/A"),
                ball.get("footwork", "N/A"),
                ball.get("shot_type", "N/A"),
                "Yes" if ball.get("aerial") else "No",
                "Yes" if ball.get("edged") else "No",
                "Yes" if ball.get("ball_missed") else "No"
            ])

        # ✅ Split the data into two parts (excluding the header row)
        header_row = bb_data[0]
        rows = bb_data[1:]
        half = len(rows) // 2 + (len(rows) % 2 > 0)  # handle odd rows

        left_table_data = [header_row] + rows[:half]
        right_table_data = [header_row] + rows[half:]

        # ✅ Create individual tables for each half
        def create_half_table(data_rows):
            t = Table(data_rows, colWidths=[23, 23, 50, 50, 43, 24, 25, 25])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            return t

        left_table = create_half_table(left_table_data)
        right_table = create_half_table(right_table_data)

        # ✅ Wrap in a single 2-column layout
        combined_table = Table([[left_table, right_table]], colWidths=[270, 270])
        combined_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6)
        ]))

        # ✅ Add to elements
        elements.append(Paragraph("<b>Ball by Ball Breakdown</b>", bold))
        elements.append(Spacer(1, 5))
        elements.append(combined_table)
        elements.append(Spacer(1, 10))

        elements.append(PageBreak())


    # 6️⃣ Detailed Bowling Summary
    if data["bowling"]["total_balls"] > 0:
        elements.append(Paragraph("<b>Detailed Bowling Summary</b>", styles['Title']))
        elements.append(Spacer(1, 6))

        # Zone Effectiveness Table
        elements.append(Paragraph("<b>Zone Effectiveness</b>", bold))
        elements.append(Spacer(1, 6))

        zone_effectiveness = data.get("zone_effectiveness", [])
        zone_table_data = [
            ["Zone", "Balls", "Runs", "Wickets", "Avg Runs/Ball", "Dot Balls", "Dot %", "Wides", "No Balls", "False Shot %"]
        ]
        for zone in zone_effectiveness:
            zone_table_data.append([
                zone["zone"],
                zone["balls"],
                zone["runs"],
                zone["wickets"],
                zone["avg_runs_per_ball"],
                zone["dot_balls"],         # new column
                f"{zone['dot_pct']}%",
                zone["wides"],             # new column
                zone["no_balls"],          # new column
                f"{zone['false_shot_pct']}%"
            ])

        zone_table = Table(zone_table_data)
        zone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        elements.append(zone_table)
        elements.append(Spacer(1, 10))

        # 6️⃣ Pitch Map Page
        if os.path.exists("/tmp/pitch_map_chart.png"):
            elements.append(Paragraph("<b>Pitch Map</b>", bold))
            elements.append(Spacer(1, 6))
            elements.append(Image("/tmp/pitch_map_chart.png", width=300, height=400))
            elements.append(Spacer(1, 6))
            add_pitch_map_legend(elements)
            elements.append(PageBreak())
        else:
            print("❌ /tmp/pitch_map_chart.png not found - skipping pitch map in PDF")


    doc.build(elements)
    buffer.seek(0)
    return buffer

def add_wagon_wheel_legend(elements):
    legend_items = [
        ("0 Runs", colors.grey),
        ("1 Run", colors.white),
        ("2 Runs", colors.yellow),
        ("3 Runs", colors.orange),
        ("4 Runs", colors.blue),
        ("5 Runs", colors.pink),
        ("6 Runs", colors.red),
    ]

    legend_flowables = []
    for label, color in legend_items:
        square = ColorSquare(color, size=8)
        legend_flowables.append(square)
        legend_flowables.append(Spacer(2, 0))

        # Use Paragraph with no-wrap style
        p = Paragraph(label, ParagraphStyle(
            name="LegendLabel",
            fontSize=8,
            leading=9,
            spaceAfter=0,
            wordWrap='CJK',
            allowOrphans=1,
            allowWidows=1,
            splitLongWords=False
        ))
        legend_flowables.append(p)

        legend_flowables.append(Spacer(8, 0))  # space between legend items

    # 🟩 Set explicit column widths for each item
    col_widths = []
    for _ in legend_items:
        col_widths.extend([10, 2, 50, 8])  # adjust widths for square, spacer, label, spacer

    # Single row table with fixed widths
    legend_table = Table([legend_flowables], colWidths=col_widths, hAlign='LEFT')
    legend_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    # 🧱 Add to PDF elements
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Wagon Wheel Legend:</b>",
        ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=9)
    ))
    elements.append(Spacer(1, 4))
    elements.append(legend_table)
    elements.append(Spacer(1, 10))

def add_pitch_map_legend(elements):
    legend_items = [
        ("Dot Ball", colors.red),
        ("Runs (1-3)", colors.green),
        ("Boundary (4/6)", colors.blue),
        ("Wicket", colors.white),
        ("Wide", colors.yellow),
        ("No Ball", colors.orange),
    ]


    legend_flowables = []
    for label, color in legend_items:
        square = ColorSquare(color, size=8)
        legend_flowables.append(square)
        legend_flowables.append(Spacer(2, 0))

        # Use Paragraph with no-wrap style
        p = Paragraph(label, ParagraphStyle(
            name="LegendLabel",
            fontSize=8,
            leading=9,
            spaceAfter=0,
            wordWrap='CJK',
            allowOrphans=1,
            allowWidows=1,
            splitLongWords=False
        ))
        legend_flowables.append(p)

        legend_flowables.append(Spacer(10, 0))  # space between legend items

    # 🟩 Set explicit column widths for each item
    col_widths = []
    for _ in legend_items:
        col_widths.extend([10, 2, 50, 8])  # adjust widths for square, spacer, label, spacer

    # Single row table with fixed widths
    legend_table = Table([legend_flowables], colWidths=col_widths, hAlign='LEFT')
    legend_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    # 🧱 Add to PDF elements
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Pitch Map Legend:</b>",
        ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=9)
    ))
    elements.append(Spacer(1, 2))
    elements.append(legend_table)

def fetch_match_summary(cursor, match_id: int, team_id: int):
    # Get innings summaries (team totals from the innings table)
    cursor.execute("""
        SELECT 
            i.innings_id,
            i.batting_team,
            i.total_runs,
            i.wickets,
            i.overs_bowled
        FROM innings i
        WHERE i.match_id = ?
        ORDER BY i.innings_id ASC
    """, (match_id,))
    innings = cursor.fetchall()

    innings_data = []
    for inn in innings:
        # Convert overs from decimal to cricket notation (e.g., 13.8333 -> 13.5)
        def convert_overs_decimal(overs_decimal: float) -> float:
            overs_int = int(overs_decimal)
            balls_fraction = overs_decimal - overs_int
            balls = int(round(balls_fraction * 6))
            return overs_int + (balls / 10)

        overs_decimal = inn["overs_bowled"] or 0
        overs_cricket = convert_overs_decimal(overs_decimal)

        # Complete Batting Scorecard
        cursor.execute("""
            SELECT p.player_name, SUM(be.runs) AS runs, 
                       COUNT(CASE WHEN (be.wides = 0 OR be.wides IS NULL) THEN 1 ELSE NULL END) AS balls
            FROM ball_events be
            JOIN players p ON be.batter_id = p.player_id
            WHERE be.innings_id = ?
            GROUP BY be.batter_id
            ORDER BY runs DESC, balls ASC
        """, (inn["innings_id"],))
        batting_card = [
            {"name": b["player_name"], "runs": b["runs"], "balls": b["balls"]}
            for b in cursor.fetchall()
        ]

        # Complete Bowling Scorecard
        cursor.execute("""
            SELECT p.player_name,
                SUM(be.runs + be.wides + be.no_balls) AS runs_conceded,
                SUM(
                    CASE
                        WHEN be.dismissed_player_id = be.batter_id
                            AND LOWER(be.dismissal_type) NOT IN ('not out', 'run out', 'obstructing the field', 'retired hurt', 'retired out')
                        THEN 1 ELSE 0
                    END
                ) AS wickets,
                COUNT(CASE 
                    WHEN (be.wides = 0 OR be.wides IS NULL) AND (be.no_balls = 0 OR be.no_balls IS NULL) THEN 1 
                    ELSE NULL END) AS balls_bowled
            FROM ball_events be
            JOIN players p ON be.bowler_id = p.player_id
            WHERE be.innings_id = ?
            GROUP BY be.bowler_id
            ORDER BY wickets DESC, runs_conceded ASC
        """, (inn["innings_id"],))

        bowling_card = []
        for b in cursor.fetchall():
            overs_bowled = convert_overs_decimal(b["balls_bowled"] / 6)  # legal deliveries
            bowling_card.append({
                "name": b["player_name"],
                "runs_conceded": b["runs_conceded"],
                "wickets": b["wickets"],
                "overs": overs_bowled
            })

        innings_data.append({
            "innings_id": inn["innings_id"],
            "batting_team": inn["batting_team"],
            "total_runs": inn["total_runs"],
            "wickets": inn["wickets"],
            "overs": overs_cricket,
            "batting_card": batting_card,
            "bowling_card": bowling_card
        })

    # Basic match info
    cursor.execute("""
        SELECT m.match_date, c1.country_name AS team_a, c2.country_name AS team_b, m.toss_winner,
               m.result
        FROM matches m
        JOIN countries c1 ON m.team_a = c1.country_id
        JOIN countries c2 ON m.team_b = c2.country_id
        WHERE m.match_id = ?
    """, (match_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "match_date": row["match_date"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "toss_winner": row["toss_winner"],
        "result": row["result"],
        "innings": innings_data
    }

def calculate_kpis(cursor, match_id: int, team_id: int, team_name: str):
    # 🎯 Medal thresholds for all KPIs
    thresholds_config = {
        "Total Runs": {"Platinum": 180, "Gold": 160, "Silver": 140, "Bronze": 120},
        "Scoring Shot %": {"Platinum": 70, "Gold": 60, "Silver": 55, "Bronze": 50},
        "PP Wickets": {"Platinum": 0, "Gold": 0, "Silver": 0, "Bronze": 1},
        "PP Runs": {"Platinum": 60, "Gold": 50, "Silver": 40, "Bronze": 30},
        "PP Boundaries": {"Platinum": 8, "Gold": 6, "Silver": 5, "Bronze": 4},
        "Death Scoring Shot %": {"Platinum": 85, "Gold": 75, "Silver": 70, "Bronze": 65},
        "Total Runs Conceded": {"Platinum": 80, "Gold": 100, "Silver": 110, "Bronze": 120},
        "Dot Ball %": {"Platinum": 70, "Gold": 60, "Silver": 55, "Bronze": 50},
        "PP Dot Ball %": {"Platinum": 75, "Gold": 70, "Silver": 65, "Bronze": 60},
        "PP Boundaries (Bowling)": {"Platinum": 0, "Gold": 2, "Silver": 3, "Bronze": 4},
        "PP Wickets (Bowling)": {"Platinum": 4, "Gold": 3, "Silver": 2, "Bronze": 1},
        "PP Score (Bowling)": {"Platinum": 15, "Gold": 20, "Silver": 25, "Bronze": 30},
        "Extras": {"Platinum": 2, "Gold": 5, "Silver": 7, "Bronze": 10},
        "Death Boundaries": {"Platinum": 0, "Gold": 2, "Silver": 3, "Bronze": 4},
        "Chances Taken %": {"Platinum": 100, "Gold": 90, "Silver": 80, "Bronze": 70},
        "Run Outs Taken %": {"Platinum": 100, "Gold": 90, "Silver": 85, "Bronze": 70},
    }

    kpis = []
    medal_tally = {
    "batting": {"Platinum": 0, "Gold": 0, "Silver": 0, "Bronze": 0},
    "bowling": {"Platinum": 0, "Gold": 0, "Silver": 0, "Bronze": 0},
    "fielding": {"Platinum": 0, "Gold": 0, "Silver": 0, "Bronze": 0},
    }

    def assign_medal(value, thresholds, lower_is_better=False):
        thresholds = thresholds.copy()
        if lower_is_better:
            thresholds = {k: -v for k, v in thresholds.items()}
            value = -value
        if value >= thresholds["Platinum"]:
            return "Platinum"
        elif value >= thresholds["Gold"]:
            return "Gold"
        elif value >= thresholds["Silver"]:
            return "Silver"
        elif value >= thresholds["Bronze"]:
            return "Bronze"
        return "None"


    # Total Runs
    cursor.execute("""
        SELECT COALESCE(SUM(be.runs), 0) + COALESCE(SUM(be.wides), 0) +
            COALESCE(SUM(be.no_balls), 0) + COALESCE(SUM(be.byes), 0) +
            COALESCE(SUM(be.leg_byes), 0) AS total_runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ?
    """, (match_id, team_name))
    actual = cursor.fetchone()["total_runs"] or 0
    thresholds = thresholds_config["Total Runs"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "Total Runs", "actual": actual, "targets": thresholds, "medal": medal})


    # Scoring Shot Pecentage
    cursor.execute("""
        SELECT COUNT(*) AS total_balls,
            SUM(CASE WHEN be.runs > 0 THEN 1 ELSE 0 END) AS scoring_shots
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ?
    """, (match_id, team_name))
    row = cursor.fetchone()
    scoring_shots = row["scoring_shots"] or 0
    total_balls = row["total_balls"] or 1  # prevent division by zero
    actual = (scoring_shots / total_balls) * 100
    thresholds = thresholds_config["Scoring Shot %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "Scoring Shot %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})

    # Powerplay Wickets
    cursor.execute("""
        SELECT COUNT(*) AS wickets
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND be.is_powerplay = 1 AND be.dismissal_type IS NOT NULL
    """, (match_id, team_name))
    actual = cursor.fetchone()["wickets"] or 0
    thresholds = thresholds_config["PP Wickets"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # Lower is better
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "PP Wickets", "actual": actual, "targets": thresholds, "medal": medal})


    # Powerplay Runs
    cursor.execute("""
        SELECT COALESCE(SUM(be.runs), 0) + COALESCE(SUM(be.wides), 0) +
            COALESCE(SUM(be.no_balls), 0) + COALESCE(SUM(be.byes), 0) + 
            COALESCE(SUM(be.leg_byes), 0) AS pp_runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND be.is_powerplay = 1
    """, (match_id, team_name))
    actual = cursor.fetchone()["pp_runs"] or 0
    thresholds = thresholds_config["PP Runs"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "PP Runs", "actual": actual, "targets": thresholds, "medal": medal})


    # Powerplay Boundaries
    cursor.execute("""
        SELECT COUNT(*) AS boundaries
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND be.is_powerplay = 1 AND be.runs >= 4
    """, (match_id, team_name))
    actual = cursor.fetchone()["boundaries"] or 0
    thresholds = thresholds_config["PP Boundaries"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "PP Boundaries", "actual": actual, "targets": thresholds, "medal": medal})


    # Partnerships >25
    cursor.execute("""
        SELECT COUNT(*) AS partnerships
        FROM partnerships p
        JOIN innings i ON p.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND p.runs >= 25
    """, (match_id, team_name))
    actual = "Yes" if cursor.fetchone()["partnerships"] >= 3 else "No"
    medal = "Gold" if actual == "Yes" else "None"
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "3x25+ Partnerships", "actual": actual, "targets": "Yes", "medal": medal})

    # Partnerships >15
    cursor.execute("""
        SELECT COUNT(*) AS partnerships
        FROM partnerships p
        JOIN innings i ON p.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND p.runs >= 15
    """, (match_id, team_name))
    actual = "Yes" if cursor.fetchone()["partnerships"] >= 2 else "No"
    medal = "Gold" if actual == "Yes" else "None"
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "2x15+ Partnerships", "actual": actual, "targets": "Yes", "medal": medal})



    # Death Scoring Shot Percentage
    cursor.execute("""
        SELECT COUNT(*) AS total_balls,
            SUM(CASE WHEN be.runs > 0 THEN 1 ELSE 0 END) AS scoring_shots
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ? AND be.over_number >= 16
    """, (match_id, team_name))
    row = cursor.fetchone()
    scoring_shots = row["scoring_shots"] or 0
    total_balls = row["total_balls"] or 1
    actual = (scoring_shots / total_balls) * 100
    thresholds = thresholds_config["Death Scoring Shot %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["batting"]: medal_tally["batting"][medal] += 1
    kpis.append({"name": "Death Scoring Shot %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})

    # Total Runs Conceded
    cursor.execute("""
        SELECT COALESCE(SUM(be.runs), 0) + COALESCE(SUM(be.wides), 0) +
            COALESCE(SUM(be.no_balls), 0) + COALESCE(SUM(be.byes), 0) +
            COALESCE(SUM(be.leg_byes), 0) AS total_runs_conceded
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ?
    """, (match_id, team_name))
    actual = cursor.fetchone()["total_runs_conceded"] or 0
    thresholds = thresholds_config["Total Runs Conceded"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # lower is better
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "Total Runs Conceded", "actual": actual, "targets": thresholds, "medal": medal})

    # Dot Ball Percentage
    cursor.execute("""
        SELECT COUNT(*) AS total_balls,
            SUM(CASE WHEN be.runs=0 AND be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS dot_balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ?
    """, (match_id, team_name))
    row = cursor.fetchone()
    dot_balls = row["dot_balls"] or 0
    total_balls = row["total_balls"] or 1
    actual = (dot_balls / total_balls) * 100
    thresholds = thresholds_config["Dot Ball %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "Dot Ball %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})


    # Powerplay Dot Ball Percentage
    cursor.execute("""
        SELECT COUNT(*) AS total_balls,
            SUM(CASE WHEN be.runs=0 AND be.wides=0 AND be.no_balls=0 THEN 1 ELSE 0 END) AS dot_balls
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND be.is_powerplay = 1
    """, (match_id, team_name))
    row = cursor.fetchone()
    dot_balls = row["dot_balls"] or 0
    total_balls = row["total_balls"] or 1
    actual = (dot_balls / total_balls) * 100
    thresholds = thresholds_config["PP Dot Ball %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "PP Dot Ball %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})

    # Powerplay Boundaries
    cursor.execute("""
        SELECT COUNT(*) AS boundaries
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND be.is_powerplay = 1 AND be.runs >= 4
    """, (match_id, team_name))
    actual = cursor.fetchone()["boundaries"] or 0
    thresholds = thresholds_config["PP Boundaries (Bowling)"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # lower is better
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "PP Boundaries (Bowling)", "actual": actual, "targets": thresholds, "medal": medal})

    # Powerplay Wickets
    cursor.execute("""
        SELECT COUNT(*) AS wickets
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND be.is_powerplay = 1 AND be.dismissal_type IS NOT NULL

    """, (match_id, team_name))
    actual = cursor.fetchone()["wickets"] or 0
    thresholds = thresholds_config["PP Wickets (Bowling)"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "PP Wickets (Bowling)", "actual": actual, "targets": thresholds, "medal": medal})


    # Powerplay Runs
    cursor.execute("""
        SELECT COALESCE(SUM(be.runs), 0) + COALESCE(SUM(be.wides), 0) +
            COALESCE(SUM(be.no_balls), 0) + COALESCE(SUM(be.byes), 0) +
            COALESCE(SUM(be.leg_byes), 0) AS pp_score
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND be.is_powerplay = 1
    """, (match_id, team_name))
    actual = cursor.fetchone()["pp_score"] or 0
    thresholds = thresholds_config["PP Score (Bowling)"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # lower is better
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "PP Score (Bowling)", "actual": actual, "targets": thresholds, "medal": medal})

    # 0s and 1s Streak
    cursor.execute("""
        SELECT be.runs, be.byes, be.leg_byes
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ?
        ORDER BY be.innings_id, be.over_number, be.ball_number
    """, (match_id, team_name))
    balls = cursor.fetchall()

    max_streak = 0
    current_streak = 0

    for ball in balls:
        outcome = ball["runs"]
        if outcome == 0 or outcome == 1 or ball["leg_byes"] == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    kpis.append({"name": "Max Dots/1s Streak", "actual": max_streak, "targets": "-", "medal": "-"})

    # Extras
    cursor.execute("""
        SELECT COALESCE(SUM(be.wides), 0) + COALESCE(SUM(be.no_balls), 0) +
            COALESCE(SUM(be.byes), 0) + COALESCE(SUM(be.leg_byes), 0) AS extras
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ?
    """, (match_id, team_name))
    actual = cursor.fetchone()["extras"] or 0
    thresholds = thresholds_config["Extras"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # lower is better
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "Extras", "actual": actual, "targets": thresholds, "medal": medal})

    # Death Boundaries 
    cursor.execute("""
        SELECT COUNT(*) AS boundaries
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND be.over_number >= 16 AND be.runs >= 4
    """, (match_id, team_name))
    actual = cursor.fetchone()["boundaries"] or 0
    thresholds = thresholds_config["Death Boundaries"]
    medal = assign_medal(-actual, {k: -v for k, v in thresholds.items()})  # lower is better
    if medal in medal_tally["bowling"]: medal_tally["bowling"][medal] += 1
    kpis.append({"name": "Death Boundaries", "actual": actual, "targets": thresholds, "medal": medal})


    # Total catch chances (2, 6, 7)
    cursor.execute("""
        SELECT COUNT(*) AS total_chances
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND bfe.event_id IN (2, 6, 7)
    """, (match_id, team_name))
    total_chances = cursor.fetchone()["total_chances"] or 0

    # Catches taken (2)
    cursor.execute("""
        SELECT COUNT(*) AS taken
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND bfe.event_id = 2
    """, (match_id, team_name))
    taken = cursor.fetchone()["taken"] or 0

    actual = (taken / total_chances) * 100 if total_chances > 0 else 0
    thresholds = thresholds_config["Chances Taken %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["fielding"]: medal_tally["fielding"][medal] += 1
    kpis.append({"name": "Catches Taken %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})

        
    # Total run out chances (3, 8)
    cursor.execute("""
        SELECT COUNT(*) AS total_chances
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND bfe.event_id IN (3, 8)
    """, (match_id, team_name))
    total_chances = cursor.fetchone()["total_chances"] or 0

    # Run outs taken (3)
    cursor.execute("""
        SELECT COUNT(*) AS taken
        FROM ball_fielding_events bfe
        JOIN ball_events be ON bfe.ball_id = be.ball_id
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ? AND bfe.event_id = 3
    """, (match_id, team_name))
    taken = cursor.fetchone()["taken"] or 0

    actual = (taken / total_chances) * 100 if total_chances > 0 else 0
    thresholds = thresholds_config["Run Outs Taken %"]
    medal = assign_medal(actual, thresholds)
    if medal in medal_tally["fielding"]: medal_tally["fielding"][medal] += 1
    kpis.append({"name": "Run Outs Taken %", "actual": round(actual, 2), "targets": thresholds, "medal": medal})


    # Continue similarly for each KPI below
    # For example:
    # - Powerplay Runs
    # - Powerplay Wickets
    # - Powerplay Boundaries
    # - Scoring Shot %...
    # - Fielding metrics...
    # - etc.

    # Note: for phase-wise KPIs (like Powerplay), include WHERE conditions like "be.is_powerplay=1"
    # For death overs, you might use "be.over_number >= 16"

    # Also add custom logic for Yes/No KPIs (like "Top 5 Bat through 15") with appropriate thresholds & medals
    # Example:
    # actual = "Yes" or "No"
    # medal = "Gold" if actual == "Yes" else "None"
    # Update medal_tally and kpis as above

    return kpis, medal_tally

def calculate_over_medals(cursor, match_id: int, team_name: str, total_overs: int = 20):
    # 1️⃣ Get the target score
    cursor.execute("""
        SELECT adjusted_target FROM matches WHERE match_id = ?
    """, (match_id,))
    row = cursor.fetchone()
    target_score = row["adjusted_target"] if row else None

    # 2️⃣ Calculate dynamic thresholds if chasing
    over_thresholds = {
        "Platinum": 12,
        "Gold": 8,
        "Silver": 7,
        "Bronze": 6
    }
    if target_score and target_score > 0:
        rr = target_score / total_overs
        over_thresholds = {
            "Platinum": rr + 6,
            "Gold": rr + 2,
            "Silver": rr + 1,
            "Bronze": rr
        }

    def assign_over_medal(runs):
        if runs >= over_thresholds["Platinum"]:
            return "Platinum"
        elif runs >= over_thresholds["Gold"]:
            return "Gold"
        elif runs >= over_thresholds["Silver"]:
            return "Silver"
        elif runs >= over_thresholds["Bronze"]:
            return "Bronze"
        return "None"

    # Batting over medals
    cursor.execute("""
        SELECT over_number, SUM(be.runs + be.wides + be.no_balls + be.byes + be.leg_byes) AS runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.batting_team = ?
        GROUP BY over_number
        ORDER BY over_number
    """, (match_id, team_name))
    batting_overs = cursor.fetchall()
    batting_over_medals = []
    for row in batting_overs:
        over_number = row["over_number"] + 1
        runs = row["runs"]
        medal = assign_over_medal(runs)
        batting_over_medals.append({
            "over": over_number,
            "runs": runs,
            "medal": medal
        })

    # Bowling over medals
    cursor.execute("""
        SELECT over_number, SUM(be.runs + be.wides + be.no_balls + be.byes + be.leg_byes) AS runs
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id = ? AND i.bowling_team = ?
        GROUP BY over_number
        ORDER BY over_number
    """, (match_id, team_name))
    bowling_overs = cursor.fetchall()
    bowling_over_medals = []
    for row in bowling_overs:
        over_number = row["over_number"] + 1
        runs = row["runs"]
        medal = assign_over_medal(runs)
        bowling_over_medals.append({
            "over": over_number,
            "runs": runs,
            "medal": medal
        })

    return {
        "batting_over_medals": batting_over_medals,
        "bowling_over_medals": bowling_over_medals
    }

def assign_medal(actual: float, thresholds: dict):
    if actual >= thresholds["Platinum"]:
        return "Platinum"
    elif actual >= thresholds["Gold"]:
        return "Gold"
    elif actual >= thresholds["Silver"]:
        return "Silver"
    elif actual >= thresholds["Bronze"]:
        return "Bronze"
    else:
        return "None"

def generate_team_pdf_report(data: dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    bold = ParagraphStyle(name='Bold', parent=normal, fontName='Helvetica-Bold', fontSize=12, leading=16)
    header = ParagraphStyle(
        name='Header', parent=normal, fontName='Helvetica-Bold', fontSize=14,
        textColor=colors.white, backColor=colors.darkblue, alignment=1, leading=18, spaceAfter=4
    )
    centered_style = ParagraphStyle(name='Center', alignment=TA_CENTER, parent=styles['Normal'])

    batting_kpi_names = [
        "Total Runs", "Scoring Shot %", "PP Wickets", "PP Runs", "PP Boundaries",
        "Top 5 Bat through 15", "3x25+ Partnerships", "2x15+ Partnerships",
        "Death Scoring Shot %", "Runs Per Over Setting", "Runs Per Over Chasing"
    ]
    bowling_kpi_names = [
        "Total Runs Conceded", "Dot Ball %", "PP Dot Ball %", "PP Boundaries (Bowling)",
        "PP Wickets (Bowling)", "PP Score (Bowling)", "Max Dots/1s Streak",
        "Extras", "Death Boundaries", "Runs Per Over Restricting", "Runs Per Over Defending"
    ]
    fielding_kpi_names = ["Chances Taken %", "Run Outs Taken %"]

    elements = []

    # Match summary header
    ms = data['match_summary']
    elements.append(Paragraph(f"<b>{ms['team_a']} vs {ms['team_b']}</b>", header))
    elements.append(Paragraph(f"Match Date: {ms.get('match_date', 'N/A')}", centered_style))
    elements.append(Paragraph(f"Toss Winner: {ms.get('toss_winner', 'N/A')}", centered_style))
    elements.append(Spacer(1, 10))

    # Scorecards
    innings_data = ms['innings']
    col_data = []
    for inn in innings_data:
        header_text = f"<b>{inn['batting_team']}</b> - {inn['total_runs']}/{inn['wickets']} ({inn['overs']} overs)"
        p_header = Paragraph(header_text, bold)

        batter_data = [["Batter", "Runs", "Balls", "Strike Rate"]]
        for b in inn['batting_card']:
            sr = round((b["runs"] / b["balls"]) * 100, 2) if b["balls"] > 0 else 0
            batter_data.append([b['name'], str(b['runs']), str(b['balls']), f"{sr:.2f}"])
        batter_table = Table(batter_data, colWidths=[150, 40, 40, 100])
        batter_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))

        bowler_data = [["Bowler", "Overs", "Runs", "Wickets"]]
        for b in inn['bowling_card']:
            bowler_data.append([b['name'], str(b['overs']), str(b['runs_conceded']), str(b['wickets'])])
        bowler_table = Table(bowler_data, colWidths=[150, 60, 60, 60])
        bowler_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))

        col_data.append([p_header, Spacer(1, 6), batter_table, Spacer(1, 6), bowler_table])

    summary_table = Table([col_data], colWidths=[doc.width / 2, doc.width / 2])
    summary_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"<b>Result: {ms['result']}</b>", header))
    elements.append(PageBreak())

    # KPI page
    elements.append(Paragraph("KEY PERFORMANCE INDICATORS (KPIs)", header))
    elements.append(Spacer(1, 10))

    def build_kpi_table(kpis):
        table = Table(kpis)
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        return table

    def horizontal_line():
        line = Table([['']], colWidths=[doc.width])
        line.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, -1), 0.25, colors.grey)]))
        return line

    # Batting KPIs
    elements.append(Paragraph("<b>Batting KPIs</b>", bold))
    batting_kpi_data = [["KPI", "Target", "Actual", "Medal"]]
    for kpi in data['kpis']:
        if kpi["name"] in batting_kpi_names:
            batting_kpi_data.append([Paragraph(kpi['name'], normal), str(kpi['targets']), str(kpi['actual']), Paragraph(f"<b>{kpi['medal']}</b>", normal)])
    elements.append(build_kpi_table(batting_kpi_data))
    elements.append(Spacer(1, 10))
    elements.append(horizontal_line())

    # Bowling KPIs
    elements.append(Paragraph("<b>Bowling KPIs</b>", bold))
    bowling_kpi_data = [["KPI", "Target", "Actual", "Medal"]]
    for kpi in data['kpis']:
        if kpi["name"] in bowling_kpi_names:
            bowling_kpi_data.append([Paragraph(kpi['name'], normal), str(kpi['targets']), str(kpi['actual']), Paragraph(f"<b>{kpi['medal']}</b>", normal)])
    elements.append(build_kpi_table(bowling_kpi_data))
    elements.append(Spacer(1, 10))
    elements.append(horizontal_line())

    # Fielding KPIs
    elements.append(Paragraph("<b>Fielding KPIs</b>", bold))
    fielding_kpi_data = [["KPI", "Target", "Actual", "Medal"]]
    for kpi in data['kpis']:
        if kpi["name"] in fielding_kpi_names:
            fielding_kpi_data.append([Paragraph(kpi['name'], normal), str(kpi['targets']), str(kpi['actual']), Paragraph(f"<b>{kpi['medal']}</b>", normal)])
    elements.append(build_kpi_table(fielding_kpi_data))
    elements.append(PageBreak())

    # New page: Medal tallies split by batting, bowling, fielding
    elements.append(Paragraph("MEDAL TALLIES BY AREA", header))
    for area, area_medals in data["medal_tallies_by_area"].items():
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"<b>{area.capitalize()} Medal Tally</b>", bold))
        area_table = Table([["Medal", "Count"]] + [[m, str(c)] for m, c in area_medals.items()])
        area_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        elements.append(area_table)

    # New page: Over medals (side by side with reversed logic for bowling)
    elements.append(PageBreak())
    elements.append(Paragraph("OVER MEDALS REPORT", header))
    elements.append(Spacer(1, 10))

    # Flip the batting and bowling innings headers to match the correct side
    batting_innings = ms['innings'][0]
    bowling_innings = ms['innings'][1]

    # These headers are for the over-by-over data
    batting_header_text = f"<b>{bowling_innings['batting_team']}</b> - {bowling_innings['total_runs']}/{bowling_innings['wickets']} ({bowling_innings['overs']} overs)"
    bowling_header_text = f"<b>{batting_innings['batting_team']}</b> - {batting_innings['total_runs']}/{batting_innings['wickets']} ({batting_innings['overs']} overs)"

    batting_header = Paragraph(batting_header_text, bold)
    bowling_header = Paragraph(bowling_header_text, bold)

    def build_over_table(over_medals, reverse=False):
        tally = {"Platinum": 0, "Gold": 0, "Silver": 0, "Bronze": 0}
        data = [["Over", "Runs", "Medal"]]
        for over in over_medals:
            medal = over["medal"]
            if reverse:
                if over["runs"] <= 0:
                    medal = "Platinum"
                elif over["runs"] <= 3:
                    medal = "Gold"
                elif over["runs"] <= 5:
                    medal = "Silver"
                else:
                    medal = "Bronze"
            over_number = str(int(over["over"]))
            data.append([over_number, str(over["runs"]), Paragraph(f"<b>{medal}</b>", normal)])
            if medal in tally:
                tally[medal] += 1
        table = Table(data, colWidths=[50, 50, 80])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        return table, tally

    batting_table, batting_tally = build_over_table(data["over_medals"]["batting_over_medals"])
    bowling_table, bowling_tally = build_over_table(data["over_medals"]["bowling_over_medals"], reverse=True)

    def build_tally_table(tally):
        data = [["Medal", "Count"]] + [[m, str(c)] for m, c in tally.items()]
        table = Table(data, colWidths=[50, 30])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        return table

    batting_tally_table = build_tally_table(batting_tally)
    bowling_tally_table = build_tally_table(bowling_tally)

    # Create two vertical columns: each has header, over table, tally table
    batting_column = [
        batting_header,
        Spacer(1, 2),
        batting_table,
        Spacer(1, 4),
        Paragraph("<b>Batting Over Medal Tally</b>", bold),
        batting_tally_table
    ]

    bowling_column = [
        bowling_header,
        Spacer(1, 2),
        bowling_table,
        Spacer(1, 4),
        Paragraph("<b>Bowling Over Medal Tally</b>", bold),
        bowling_tally_table
    ]

    # Put them side by side
    innings_tables = Table([[batting_column, bowling_column]], colWidths=[doc.width / 2, doc.width / 2])
    innings_tables.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(innings_tables)

    doc.build(elements)
    buffer.seek(0)
    return buffer






if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
