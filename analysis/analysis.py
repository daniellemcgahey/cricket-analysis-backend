import sqlite3
from collections import defaultdict

def get_country_stats(country, tournaments, selected_stats, selected_phases, bowler_type=None, bowling_arm=None):
        conn = sqlite3.connect('cricket_analysis.db')
        c = conn.cursor()
        
        #print(f"\nFetching stats for {country} in tournaments: {tournaments}")
        
        # Get country ID
        c.execute("SELECT country_id FROM countries WHERE country_name = ?", (country,))
        country_result = c.fetchone()
        if not country_result:
            return defaultdict(lambda: defaultdict(float))
        country_id = country_result[0]
        
        bowler_type_filter = bowler_type
        bowling_arm_filter = bowling_arm

        filters = []
        if bowler_type_filter == "Pace":
            filters.append("bowling_style = 'Pace'")
        elif bowler_type_filter == "Medium":
            filters.append("bowling_style = 'Medium'")
        elif bowler_type_filter == "Spin":
            filters.append("bowling_style = 'Spin'")
        elif bowler_type_filter == "Pace + Medium":
            filters.append("bowling_style IN ('Pace', 'Medium')")

        if bowling_arm_filter == "Left":
            filters.append("bowling_arm = 'Left'")
        elif bowling_arm_filter == "Right":
            filters.append("bowling_arm = 'Right'")

        # Join filters into a WHERE clause string (e.g. "AND bowling_style = 'Spin' AND bowling_arm = 'Left'")
        combined_filter = ""
        if filters:
            combined_filter = " AND " + " AND ".join(filters)


        # Get tournament IDs first
        c.execute(f"SELECT tournament_id FROM tournaments WHERE tournament_name IN ({','.join(['?']*len(tournaments))})", tournaments)
        tournament_ids = [row[0] for row in c.fetchall()]



        bowler_type_conditions = {
            "All": "",  # No filter
            "Pace": "AND p.bowling_style = 'Pace'",
            "Medium": "AND p.bowling_style = 'Medium'",
            "Spin": "AND p.bowling_style = 'Spin'",
            "Pace + Medium": "AND p.bowling_style IN ('Pace', 'Medium')"
        }
        bowler_condition = bowler_type_conditions.get(bowler_type, "")

        # Get matches where country participated in these tournaments
        query = f"""
        SELECT match_id 
        FROM matches 
        WHERE tournament_id IN ({','.join(['?']*len(tournament_ids))})
        AND (team_a = ? OR team_b = ?)
        """
        params = tournament_ids + [country_id, country_id]

        #print(f"Match query: {query}")
        #print(f"Params: {params}")
        
        c.execute(query, params)
        match_ids = [row[0] for row in c.fetchall()]
        #print(f"Found matches: {match_ids}")
        
        stats = {
            'batting': defaultdict(float),
            'bowling': defaultdict(float),
            'fielding': defaultdict(float)
        }
        
        if not match_ids:
            return stats

        # REVISED BATTING QUERY
        batting_query = f"""
        SELECT
            COUNT(DISTINCT be.batter_id) AS innings,
            SUM(be.runs) AS total_runs,
            COUNT(*) AS balls_faced,
            SUM(CASE WHEN be.runs = 0 AND be.extras = 0 THEN 1 ELSE 0 END) AS dot_balls,
            SUM(CASE WHEN be.runs = 1 THEN 1 ELSE 0 END) AS ones,
            SUM(CASE WHEN be.runs = 2 THEN 1 ELSE 0 END) AS twos,
            SUM(CASE WHEN be.runs = 3 THEN 1 ELSE 0 END) AS threes,
            SUM(CASE WHEN be.runs = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN be.runs = 6 THEN 1 ELSE 0 END) AS sixes,
            SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS dismissals,
            SUM(CASE WHEN LOWER(be.shot_type) = 'attacking' THEN 1 ELSE 0 END),
            SUM(CASE WHEN LOWER(be.shot_type) = 'defensive' THEN 1 ELSE 0 END),
            SUM(CASE WHEN LOWER(be.shot_type) = 'rotation' THEN 1 ELSE 0 END)
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id IN ({','.join(['?']*len(match_ids))})
        AND be.batter_id IN (
            SELECT player_id FROM players WHERE country_id = ?
        )
        AND be.bowler_id IN (
            SELECT player_id FROM players WHERE 1=1 {combined_filter}
        )
        """
        c.execute(batting_query, match_ids + [country_id])
        batting_data = c.fetchone()
        
        if batting_data:
            stats['batting']['Innings'] = batting_data[0] or 0
            stats['batting']['Runs'] = batting_data[1] or 0
            stats['batting']['Balls Faced'] = batting_data[2] or 0
            stats['batting']['Dot Balls Faced'] = batting_data[3] or 0
            stats['batting']['1s'] = batting_data[4] or 0
            stats['batting']['2s'] = batting_data[5] or 0
            stats['batting']['3s'] = batting_data[6] or 0
            stats['batting']['4s'] = batting_data[7] or 0
            stats['batting']['6s'] = batting_data[8] or 0
            stats['batting']['Dismissals'] = batting_data[9] or 0
            
            # Calculate derived stats
            if stats['batting']['Balls Faced'] > 0:
                stats['batting']['Strike Rate'] = (stats['batting']['Runs'] / stats['batting']['Balls Faced']) * 100
                stats['batting']['Dot Ball %'] = (stats['batting']['Dot Balls Faced'] / stats['batting']['Balls Faced']) * 100
            if stats['batting']['Dismissals'] > 0:
                stats['batting']['Average'] = stats['batting']['Runs'] / stats['batting']['Dismissals']
            else:
                stats['batting']['Average'] = stats['batting']['Runs']  # Not out average

                    # Batting Intent overall
            total_intent = sum(filter(None, [batting_data[10], batting_data[11], batting_data[12]]))
            if total_intent > 0:
                stats['batting']['Attacking Shot %'] = (batting_data[10] / total_intent) * 100
                stats['batting']['Defensive Shot %'] = (batting_data[11] / total_intent) * 100
                stats['batting']['Rotation Shot %'] = (batting_data[12] / total_intent) * 100

 
        # Phase-specific batting stats
        phase_conditions = {
            'Powerplay': 'is_powerplay = 1',
            'Middle Overs': 'is_middle_overs = 1',
            'Death Overs': 'is_death_overs = 1'
        }
        
        for phase, condition in phase_conditions.items():
       
            phase_query = f"""
            SELECT
                SUM(be.runs),
                COUNT(*),
                SUM(CASE WHEN be.runs = 0 AND be.extras = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN be.runs IN (4,6) THEN 1 ELSE 0 END),
                SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(be.shot_type) = 'attacking' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(be.shot_type) = 'defensive' THEN 1 ELSE 0 END),
                SUM(CASE WHEN LOWER(be.shot_type) = 'rotation' THEN 1 ELSE 0 END)
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id IN ({','.join(['?']*len(match_ids))})
            AND be.batter_id IN (
                SELECT player_id FROM players WHERE country_id = ?
            )
            AND be.bowler_id IN (
                SELECT player_id FROM players WHERE 1=1 {combined_filter}
            )
            AND {condition}
            """
            c.execute(phase_query, match_ids + [country_id])
            phase_data = c.fetchone()

            if phase_data:
                stats['batting'][f'{phase} Runs'] = phase_data[0] or 0
                stats['batting'][f'{phase} Balls'] = phase_data[1] or 0
                stats['batting'][f'{phase} Dot Balls'] = phase_data[2] or 0
                stats['batting'][f'{phase} Boundaries'] = phase_data[3] or 0
                stats['batting'][f'{phase} Dismissals'] = phase_data[4] or 0

                if phase_data[1]:
                    stats['batting'][f'{phase} Strike Rate'] = (phase_data[0] / phase_data[1]) * 100
                    stats['batting'][f'{phase} Dot %'] = (phase_data[2] / phase_data[1]) * 100

                # Batting Intent Calculations
                total_intent = sum(filter(None, [phase_data[5], phase_data[6], phase_data[7]]))
                if total_intent > 0:
                    stats['batting'][f'{phase} Attacking Shot %'] = (phase_data[5] / total_intent) * 100
                    stats['batting'][f'{phase} Defensive Shot %'] = (phase_data[6] / total_intent) * 100
                    stats['batting'][f'{phase} Rotation Shot %'] = (phase_data[7] / total_intent) * 100


        # Corrected Bowling Query
        bowling_query = f"""
        SELECT
            COUNT(*) AS balls,
            SUM(be.runs) AS runs_conceded,
            SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END) AS wickets,
            SUM(be.dot_balls) AS dot_balls,
            SUM(be.wides + be.no_balls) AS extras,
            SUM(CASE WHEN be.runs IN (4,6) THEN 1 ELSE 0 END) AS boundaries
        FROM ball_events be
        JOIN innings i ON be.innings_id = i.innings_id
        WHERE i.match_id IN ({','.join(['?']*len(match_ids))})
        AND be.bowler_id IN (
        SELECT p.player_id FROM players p WHERE p.country_id = ? {combined_filter}
        )
        """
        c.execute(bowling_query, match_ids + [country_id])
        bowling_data = c.fetchone()
        
        if bowling_data:
            stats['bowling']['Overs'] = f"{bowling_data[0]//6}.{bowling_data[0]%6}"
            stats['bowling']['Runs'] = bowling_data[1]
            stats['bowling']['Wickets'] = bowling_data[2]
            stats['bowling']['Dot Balls'] = bowling_data[3]
            stats['bowling']['Extras'] = bowling_data[4]
            stats['bowling']['Boundaries Conceded'] = bowling_data[5]
            
            if bowling_data[0] > 0:
                stats['bowling']['Economy'] = (bowling_data[1] / (bowling_data[0]/6))
                stats['bowling']['Dot Ball %'] = (bowling_data[3] / bowling_data[0]) * 100
                if bowling_data[2] > 0:
                    stats['bowling']['Average'] = bowling_data[1] / bowling_data[2]

        # Phase-specific bowling stats
        for phase, condition in phase_conditions.items():
            phase_query = f"""
            SELECT
                COUNT(*),
                SUM(be.runs),
                SUM(CASE WHEN be.dismissal_type IS NOT NULL THEN 1 ELSE 0 END),
                SUM(be.dot_balls)
            FROM ball_events be
            JOIN innings i ON be.innings_id = i.innings_id
            WHERE i.match_id IN ({','.join(['?']*len(match_ids))})
            AND be.bowler_id IN (
            SELECT p.player_id FROM players p WHERE p.country_id = ? {combined_filter}
            )
            AND {condition}
            """
            c.execute(phase_query, match_ids + [country_id])
            phase_data = c.fetchone()
            
            if phase_data:
                stats['bowling'][f'{phase} Overs'] = f"{phase_data[0]//6}.{phase_data[0]%6}"
                stats['bowling'][f'{phase} Runs'] = phase_data[1]
                stats['bowling'][f'{phase} Wickets'] = phase_data[2]
                stats['bowling'][f'{phase} Dot Balls'] = phase_data[3]
                
                if phase_data[0] > 0:
                    stats['bowling'][f'{phase} Economy'] = (phase_data[1] / (phase_data[0]/6))
                    stats['bowling'][f'{phase} Dot %'] = (phase_data[3] / phase_data[0]) * 100

                    # REVISED FIELDING QUERY
                    # -----------------------------------------
            # 🧤 FULL FIELDING EVENTS (TEAM-LEVEL)
            # -----------------------------------------
            
                # --- FIELDING IMPACT RATING (IR) ---
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
            total_ir = 0
            
            
            fielding_event_query = f"""
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
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND p.country_id = ?
            """
            c.execute(fielding_event_query, match_ids + [country_id])
            event_data = c.fetchone()

            event_names = [
                'Catch', 'Run Out', 'Drop Catch', 'Boundary Save',
                'Clean Stop/Pick Up', 'Direct Hit', 'Missed Catch', 'Missed Run Out',
                'Fumble', 'Missed Fielding', 'Overthrow', 'Taken Half Chance', 'Missed Half Chance'
            ]
            for name, count in zip(event_names, event_data):
                stats['fielding'][name] = count or 0

            # -----------------------------------------
            # 🎯 TOTAL BALLS FIELDED (TEAM-LEVEL)
            # -----------------------------------------
            balls_fielded_query = f"""
            SELECT COUNT(DISTINCT bfe.ball_id)
            FROM ball_fielding_events bfe
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players p ON be.fielder_id = p.player_id
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND p.country_id = ?
            """
            c.execute(balls_fielded_query, match_ids + [country_id])
            stats['fielding']['Total Balls Fielded'] = c.fetchone()[0] or 0

            # Expected vs Actual Runs (Fielding)
            expected_actual_query = f"""
            SELECT 
                COALESCE(SUM(be.expected_runs), 0),
                COALESCE(SUM(be.runs + be.extras + be.wides + be.no_balls + be.byes + be.leg_byes + be.penalty_runs), 0)
            FROM ball_fielding_events bfe
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players p ON be.fielder_id = p.player_id
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND p.country_id = ?
            """
            c.execute(expected_actual_query, match_ids + [country_id])
            expected_runs, actual_runs = c.fetchone()
            stats['fielding']['Expected Runs'] = expected_runs or 0
            stats['fielding']['Actual Runs'] = actual_runs or 0
            stats['fielding']['Runs Saved/Allowed'] = expected_runs - actual_runs

            # -----------------------------------------
            # 📊 TEAM FIELDING METRICS (TEAM-LEVEL)
            # -----------------------------------------
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


            opportunities = c_ + d_ + mc_ + r_ + mru_ + 0.5*thc_ + 0.5*mhc_
            successful = c_ + r_ + thc_
            stats['fielding']['Conversion Rate'] = (
                (successful / opportunities) * 100 if opportunities > 0 else 0
            )

            stats['fielding']['Pressure Score'] = (
                dh_ + cs_ + b_ - o_ - mf_ - f_
            )

            for event, weight in fielding_weights.items():
                count = stats['fielding'].get(event, 0)
                total_ir += count * weight
            stats['fielding']['Fielding Impact Rating'] = total_ir


         # Phase-specific fielding stats
        for phase, condition in phase_conditions.items():
            phase_ir = 0
                    # -----------------------------------------
            # 🧤 PHASE FIELDING EVENTS
            # -----------------------------------------
            phase_fielding_query = f"""
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
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND {condition}
            AND p.country_id = ?
            """
            c.execute(phase_fielding_query, match_ids + [country_id])
            phase_data = c.fetchone()

            event_names = [
                'Catch', 'Run Out', 'Drop Catch', 'Boundary Save',
                'Clean Stop/Pick Up', 'Direct Hit', 'Missed Catch', 'Missed Run Out',
                'Fumble', 'Missed Fielding', 'Overthrow', 'Taken Half Chance', 'Missed Half Chance'
            ]
            for name, count in zip(event_names, phase_data):
                stats['fielding'][f"{phase} {name}"] = count or 0

            # -----------------------------------------
            # 🎯 TOTAL BALLS FIELDED (PHASE)
            # -----------------------------------------
            phase_ball_query = f"""
            SELECT COUNT(DISTINCT bfe.ball_id)
            FROM ball_fielding_events bfe
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players p ON be.fielder_id = p.player_id
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND {condition}
            AND p.country_id = ?
            """
            c.execute(phase_ball_query, match_ids + [country_id])
            stats['fielding'][f"{phase} Total Balls Fielded"] = c.fetchone()[0] or 0

            # Expected vs Actual Runs (Phase)
            phase_exp_actual_query = f"""
            SELECT 
                COALESCE(SUM(be.expected_runs), 0),
                COALESCE(SUM(be.runs + be.extras + be.wides + be.no_balls + be.byes + be.leg_byes + be.penalty_runs), 0)
            FROM ball_fielding_events bfe
            JOIN ball_events be ON bfe.ball_id = be.ball_id
            JOIN innings i ON be.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            JOIN players p ON be.fielder_id = p.player_id
            WHERE m.match_id IN ({','.join(['?']*len(match_ids))})
            AND {condition}
            AND p.country_id = ?
            """
            c.execute(phase_exp_actual_query, match_ids + [country_id])
            phase_exp, phase_actual = c.fetchone()
            stats['fielding'][f"{phase} Expected Runs"] = phase_exp or 0
            stats['fielding'][f"{phase} Actual Runs"] = phase_actual or 0
            stats['fielding'][f"{phase} Runs Saved/Allowed"] = phase_exp - phase_actual

            # -----------------------------------------
            # 📊 PHASE FIELDING METRICS
            # -----------------------------------------
            c_ = stats['fielding'][f"{phase} Catch"]
            r_ = stats['fielding'][f"{phase} Run Out"]
            d_ = stats['fielding'][f"{phase} Drop Catch"]
            b_ = stats['fielding'][f"{phase} Boundary Save"]
            cs_ = stats['fielding'][f"{phase} Clean Stop/Pick Up"]
            dh_ = stats['fielding'][f"{phase} Direct Hit"]
            mc_ = stats['fielding'][f"{phase} Missed Catch"]
            mru_ = stats['fielding'][f"{phase} Missed Run Out"]
            f_ = stats['fielding'][f"{phase} Fumble"]
            mf_ = stats['fielding'][f"{phase} Missed Fielding"]
            o_ = stats['fielding'][f"{phase} Overthrow"]
            thc_ = stats['fielding'][f"{phase} Taken Half Chance"]
            mhc_ = stats['fielding'][f"{phase} Missed Half Chance"]


            opportunities = c_ + d_ + mc_ + r_ + mru_ + 0.5*thc_ + 0.5*mhc_
            successful = c_ + r_ + thc_
            stats['fielding'][f"{phase} Conversion Rate"] = (
                (successful / opportunities) * 100 if opportunities > 0 else 0
            )

            stats['fielding'][f"{phase} Pressure Score"] = (
                dh_ + cs_ + b_ - o_ - mf_ - f_
            )

            for event, weight in fielding_weights.items():
                key = f"{phase} {event}"
                count = stats['fielding'].get(key, 0)
                phase_ir += count * weight
            stats['fielding'][f'{phase} Fielding Impact Rating'] = phase_ir

        conn.close()
        return stats