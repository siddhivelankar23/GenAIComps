# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
export service_name="text2kg"
export LOG_PATH=$PWD
docker compose -f compose.yaml -f custom-override.yml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
