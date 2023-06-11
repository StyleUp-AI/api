import datetime
import os
from typing import Any, List, Optional, Union

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class GoogleCalendarReader(BaseReader):
    """Google Calendar reader.

    Reads events from Google Calendar

    """

    def load_data(
        self,
        token,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
    ):

        """Load data from user's calendar.

        Args:
            number_of_results (Optional[int]): the number of events to return. Defaults to 100.
            start_date (Optional[Union[str, datetime.date]]): the start date to return events from. Defaults to today.
        """

        from googleapiclient.discovery import build

        credentials = self._get_credentials(token)
        if credentials == 'Need to login to google':
            return credentials
        service = build("calendar", "v3", credentials=credentials)

        if start_date is None:
            start_date = datetime.date.today()
        elif isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)

        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        start_datetime_utc = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_datetime_utc,
                maxResults=number_of_results,
                singleEvents=False,
            )
            .execute()
        )

        events = events_result.get("items", [])

        if not events:
            return []

        results = []
        for event in events:
            if "dateTime" in event["start"]:
                start_time = event["start"]["dateTime"]
            else:
                start_time = event["start"]["date"]

            if "dateTime" in event["end"]:
                end_time = event["end"]["dateTime"]
            else:
                end_time = event["end"]["date"]

            event_string = f"Status: {event['status']}, "
            event_string += f"Summary: {event['summary']}, "
            event_string += f"Start time: {start_time}, "
            event_string += f"End time: {end_time}, "

            organizer = event.get("organizer", {})
            display_name = organizer.get("displayName", "N/A")
            email = organizer.get("email", "N/A")
            if display_name != "N/A":
                event_string += f"Organizer: {display_name} ({email})"
            else:
                event_string += f"Organizer: {email}"

            results.append(Document(event_string))

        return results

    def _get_credentials(self, token=None) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        if token is None:
            return 'Need to login to google'
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

        creds = Credentials(token, 
                            efresh_token=token, 
                            token_uri="https://oauth2.googleapis.com/token", 
                            client_id="174069416578-fgb8ks6su101kh793nduk9uqn03u9jpd.apps.googleusercontent.com", 
                            client_secret="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", 
                            scopes=SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            return 'Need to login to google'

        return creds