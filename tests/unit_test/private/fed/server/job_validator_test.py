# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
from typing import Dict, Tuple, Optional, List

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContextManager, FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci import zip_utils
from nvflare.private.fed.server.job_validator import JobValidator
from nvflare.widgets.widget import Widget


class MockServerEngine(ServerEngineSpec):

    def __init__(self):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            run_num="unit-test-run",
            public_stickers={},
            private_stickers={}
        )

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass

    def get_clients(self) -> List[Client]:
        pass

    def sync_clients_from_main_process(self):
        pass

    def validate_clients(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        pass

    def new_context(self) -> FLContext:
        return self.fl_ctx_mgr.new_context()

    def get_workspace(self) -> Workspace:
        pass

    def get_component(self, component_id: str) -> object:
        pass

    def register_aux_message_handler(self, topic: str, message_handle_func):
        pass

    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        pass

    def get_widget(self, widget_id: str) -> Widget:
        pass

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        pass

    def restore_components(self, snapshot: RunSnapshot, fl_ctx: FLContext):
        pass

    def start_client_job(self, run_number, client_sites):
        pass

    def check_client_resources(self, resource_reqs: Dict[str, dict]) -> Dict[str, Tuple[bool, Optional[str]]]:
        pass

    def cancel_client_resources(self, resource_check_results: Dict[str, Tuple[bool, str]],
                                resource_reqs: Dict[str, dict]):
        pass

    def get_client_name_from_token(self, token: str) -> str:
        pass


class TestJobValidator:

    @classmethod
    def setup_class(cls):
        cls.engine = MockServerEngine()
        cls.fl_ctx = cls.engine.new_context()
        cls.validator = JobValidator(cls.fl_ctx)

    def test_valid_app(self):
        self._assert_valid("valid_app")

    def test_valid_job(self):
        self._assert_valid("valid_job")

    def test_not_enough_clients(self):
        self._assert_invalid("not_enough_clients")

    def test_duplicate_server(self):
        self._assert_invalid("duplicate_server")

    def test_duplicate_clients(self):
        self._assert_invalid("duplicate_clients")

    def test_mandatory_not_met(self):
        self._assert_invalid("mandatory_not_met")

    def test_missing_app(self):
        self._assert_invalid("missing_app")

    def test_missing_server_config(self):
        self._assert_invalid("missing_server_config")

    def test_missing_client_config(self):
        self._assert_invalid("missing_client_config")

    def _assert_valid(self, job_name: str):
        data = self._zip_job(job_name)
        valid, error, meta = self.validator.validate(job_name, data)
        assert valid, error

    def _assert_invalid(self, job_name: str):
        data = self._zip_job(job_name)
        valid, error, meta = self.validator.validate(job_name, data)
        assert not valid, error

    @staticmethod
    def _zip_job(job_name: str) -> bytes:
        job_path = os.path.join(os.path.dirname(__file__), "../../../data/jobs")
        zip_data = zip_utils.zip_directory_to_bytes(job_path, job_name)
        return zip_utils.convert_legacy_zip(zip_data)