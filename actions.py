from celery import Celery
import subprocess
import subprocess
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import EventType, ReminderScheduled, SlotSet
from typing import Any, Text, Dict, List
from datetime import datetime, timedelta
from celery_app import run_llama
import subprocess
from transformers import BertModel, BertTokenizer
import torch
from pymilvus import Collection, connections, utility
import numpy as np
from sentence_transformers import SentenceTransformer
from rasa_sdk.events import SlotSet, Restarted, AllSlotsReset, UserUttered, FollowupAction, SessionStarted, ActionExecuted, UserUtteranceReverted
import psycopg2
import telegram
from telegram import Bot

MINUTES = 3

# Database connection
def connect_to_db():
    conn = psycopg2.connect(
        dbname="rasa",
        user="rasauser",
        password="password",
        host="localhost"
    )
    return conn

class ActionInitializeConversation(Action):
    def name(self):
        return "initialize_conversation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_id = tracker.sender_id
        user_name = None
        first_name = None
        last_name = None

        text = tracker.latest_message.get('text')

        # Extract user_id and user_name from the start link
        if tracker.get_latest_input_channel() == 'telegram' or text == "/start":
            # Use Telegram API to fetch the username
            TELEGRAM_TOKEN = "6520643484:AAGHlm5vt4llZFAPTlSn_fjlLznrB72avp8"
            bot = Bot(token=TELEGRAM_TOKEN)
            try:
                user = bot.get_chat(user_id)
                user_name = user.username
                first_name = user.first_name
                last_name = user.last_name
            except Exception as e:
                print(f"Error fetching user info from Telegram: {e}")

        conn = connect_to_db()
        cursor = conn.cursor()

        # Fetch or initialize conversation count for the user
        cursor.execute("SELECT conversation_count FROM users WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            conversation_count = result[0] + 1
            cursor.execute(
                "UPDATE users SET conversation_count = %s, user_name = %s, first_name = %s, last_name = %s WHERE user_id = %s",
                (conversation_count, user_name, first_name, last_name, user_id)
            )
        else:
            conversation_count = 0
            cursor.execute(
                "INSERT INTO users (user_id, user_name, first_name, last_name, conversation_count) VALUES (%s, %s, %s, %s, %s)",
                (user_id, user_name, first_name, last_name, conversation_count)
            )

        # Create a new conversation entry and get its ID
        cursor.execute("""
            INSERT INTO conversations (
                user_id, conversation_count, conversation, selected_option
            ) VALUES (%s, %s, %s, %s) RETURNING id
        """, (user_id, conversation_count, "", None))
        
        conversation_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        return [SlotSet("conversation_id", conversation_id)]

# Other actions remain unchanged


class ActionDefaultFallback(Action):
    def name(self):
        return "action_default_fallback"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(response="utter_confused")
        # Optionally, you can use `UserUtteranceReverted()` to ignore the last user message
        # and give the user another chance to rephrase it:
        return [] #UserUtteranceReverted()


class ActionReset(Action):
    def name(self):
        return "reset-and-restart"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        user_id = tracker.sender_id
        
        # Collect both user and bot events
        conversation_parts = []
        for e in tracker.events:
            if e.get("event") in ["user", "bot"]:
                text = e.get("text")
                if text:
                    conversation_parts.append(text)
        
        conversation = "\n".join(conversation_parts)
        conversation_id = tracker.get_slot('conversation_id')

        conn = connect_to_db()
        cursor = conn.cursor()

        # Update the conversation with the full text
        cursor.execute("""
            UPDATE conversations 
            SET conversation = %s 
            WHERE id = %s
        """, (conversation, conversation_id))

        # Fetch the updated conversation count for the user
        cursor.execute("SELECT conversation_count FROM users WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            conversation_count = result[0]
        else:
            conversation_count = 0  # Shouldn't happen since user should already be initialized

        conn.commit()
        cursor.close()
        conn.close()

        # Utter the message with the conversation count
        dispatcher.utter_message(text=f"Conversation ended ({conversation_count}/3)")

        return [Restarted(), AllSlotsReset()]


class ActionStoreActionPrompt(Action):
    def name(self) -> str:
        return "store-action-prompt"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Extract the full message text from the last user message
        text = tracker.latest_message.get('text')

        # You may want to do some processing or logging here

        # Store the text in a slot
        return [SlotSet("action_prompt", text)]

class ActionStoreSelectedAction(Action):
    def name(self) -> str:
        return "store-selected-action"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Extract the full message text from the last user message)
        
        text = tracker.latest_message.get('text')
        # You may want to do some processing or logging here

        # Store the text in a slot
        return [SlotSet("selected_option", text)]

class ActionHandleSelectedOption(Action):
    def name(self) -> Text:
        return "action-handle-selected-option"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        selected_option = tracker.get_slot('selected_option')
        conversation_id = tracker.get_slot('conversation_id')

        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations 
            SET selected_option = %s 
            WHERE id = %s
        """, (selected_option, conversation_id))
        conn.commit()
        cursor.close()
        conn.close()

        message = f"You selected: {selected_option}"
        dispatcher.utter_message(text=message)
        return []


class ActionReportProblem(Action):
    def name(self) -> Text:
        return "report-problem"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        problem = tracker.get_slot('action_prompt')
        conversation_id = tracker.get_slot('conversation_id')

        problem_id = self.main(problem)

        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reported_problems (conversation_id, problem_id, problem_text)
            VALUES (%s, %s, %s)
        """, (conversation_id, problem_id, problem))
        conn.commit()
        cursor.close()
        conn.close()

        message = f"'{problem}' anonymously added to the database, other members of your neighbourhood will help to solve this issue."
        dispatcher.utter_message(text=message)
        return []

    def main(self, problem):
        command = ["python3", "/home/orlando/action-scripts/store-search-embeddings.py", "insert", problem]

        try:
            output = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            problem_id = int(output.stdout.strip()[1:-1])
            print(problem_id)
            return problem_id
        except subprocess.CalledProcessError as e:
            print(f"Failed to run script: {e}")
            return None

class ActionSearchProblem(Action):
    def name(self) -> Text:
        return "search-problem"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        skill = tracker.get_slot('action_prompt')
        conversation_id = tracker.get_slot('conversation_id')

        distance, problem_text, problem_id = self.main(skill)

        if distance <= 1.3:
            buttons = [{"title": "Yes.", "payload": "/positive"}, {"title": "No.", "payload": "/negative"}]
            message = f"I found the following problem from a member of your neighbourhood that may be related: '{problem_text}'. Is this something you're interested in working on a solution?"
            dispatcher.utter_message(text=message, buttons=buttons)

            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO searched_problems (conversation_id, problem_id, problem_text)
                VALUES (%s, %s, %s)
            """, (conversation_id, problem_id, problem_text))
            conn.commit()
            cursor.close()
            conn.close()

            return [SlotSet("neighbour_problem_text", problem_text)]
        else:
            return [FollowupAction(name='llama-action')]

    def main(self, skill):
        command = ["python3", "/home/orlando/action-scripts/store-search-embeddings.py", "search", skill]

        try:
            output = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            problem_id, distance, text = output.stdout.split(", ")
            print(problem_id)
            return float(distance), str(text), int(problem_id)
        except subprocess.CalledProcessError as e:
            print(f"Failed to run script: {e}")
            return 0, "", None

#informed llama
class ActionHandleLlamaInformationReminder(Action):

    def name(self) -> Text:
        return "handle-llama-action-response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        task_id = tracker.get_slot("task_id")
        actions_classified = tracker.get_slot("actions_classified")

        result = run_llama.AsyncResult(task_id)
        
        if result.ready():
            response = result.result
            buttons = self.parse_buttons(actions_classified)
            dispatcher.utter_message(text=response, buttons=buttons)
        else:
            dispatcher.utter_message(text="Error, response was not ready. Server overloaded, please come back later.")
            [FollowupAction(name='reset-and-restart')]

        return []
    
    def parse_buttons(self, action_response):
        options = action_response.split(',')
        buttons = []
        for option in options:
            option_text = option.strip()
            buttons.append({"title": option_text, "payload": option_text})
        return buttons

class ActionHandleLlamaInformationReminder(Action):

    def name(self) -> Text:
        return "handle-llama-informed-response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        task_id = tracker.get_slot("task_id")
        result = run_llama.AsyncResult(task_id)
        
        if result.ready():
            response = result.result
            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(text="Error, response was not ready. Server overloaded, please come back later.")
            [FollowupAction(name='reset-and-restart')]

        return []

class ActionLlamaCapability(Action):

    def name(self) -> Text:
        return "llama-capability"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)

        # Get the user's latest message
        latest_message = tracker.latest_message.get('text')
        task = run_llama.delay("capability", latest_message, "")
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="informed_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("task_id", task.id), reminder]

class ActionLlamaInformation(Action):

    def name(self) -> Text:
        return "llama-information"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)

        # Get the user's latest message
        latest_message = tracker.latest_message.get('text')
        task = run_llama.delay("information", latest_message, "")
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="informed_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("task_id", task.id), reminder]
    
class ActionLlamaProblem(Action):

    def name(self) -> Text:
        return "llama-problem"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)

        # Get the user's latest message
        latest_message = tracker.latest_message.get('text')
        task = run_llama.delay("problem", latest_message, "")
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="informed_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("task_id", task.id), reminder]

class ActionLlamaLoneliness(Action):

    def name(self) -> Text:
        return "llama-loneliness"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)

        # Get the user's latest message
        latest_message = tracker.latest_message.get('text')
        task = run_llama.delay("loneliness", latest_message, "")
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="informed_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("task_id", task.id), reminder]

class ActionLlamaDisconnection(Action):

    def name(self) -> Text:
        return "llama-disconnection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)

        # Get the user's latest message
        latest_message = tracker.latest_message.get('text')
        task = run_llama.delay("disconnection", latest_message, "")
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="informed_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("task_id", task.id), reminder]

#action llama

class ActionLlamaAction(Action):
    def name(self) -> str:
        return "llama-action"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Retrieve the slot value
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)
        action_prompt = tracker.get_slot('action_prompt')

        action_response = self.classifier(action_prompt)
        
        task = run_llama.delay("action", action_prompt, action_response)
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="action_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("actions_classified", action_response),SlotSet("task_id", task.id), reminder]
    
    def classifier(self, action_prompt):
        command = ["python3", "/home/orlando/action-scripts/action-classifier-inference.py", action_prompt]
        try:
            output = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            response = output.stdout

    
        except subprocess.CalledProcessError as e:
            response = f"Failed to run script: {e}"
        
        return response

class ActionLlamaActionProblemSkill(Action):
    def name(self) -> str:
        return "llama-action-problem-skill"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Retrieve the slot value
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)
        action_prompt = tracker.get_slot('action_prompt')

        neighbour_problem = tracker.get_slot('neighbour_problem_text')

        action_response = self.classifier(action_prompt)

        prompt = f"{action_prompt}. I want to solve the problem of another neighbour who stated: '{neighbour_problem}'"
        
        task = run_llama.delay("action", prompt, action_response)
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="action_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("actions_classified", action_response),SlotSet("task_id", task.id), reminder]
    
    def classifier(self, action_prompt):
        command = ["python3", "/home/orlando/action-scripts/action-classifier-inference.py", action_prompt]
        try:
            output = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            response = output.stdout
    
        except subprocess.CalledProcessError as e:
            response = f"Failed to run script: {e}"
        
        return response

class ActionLlamaActionNoSkill(Action):
    def name(self) -> str:
        return "llama-action-noskill"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Retrieve the slot value
        reminder_time = datetime.now() + timedelta(minutes=MINUTES)
        action_prompt = "I have no skills or interests"

        action_response = "Start/join a group, Share a story"

        task = run_llama.delay("action", action_prompt, action_response)
        dispatcher.utter_message(text="Your request is being processed, please wait... (around 3 minutes)")

        reminder = ReminderScheduled(
            intent_name="action_response_intent",
            trigger_date_time=reminder_time,
            name="llama_task_reminder",
            kill_on_user_message=False
        )

        return [SlotSet("actions_classified", action_response),SlotSet("task_id", task.id), reminder]
