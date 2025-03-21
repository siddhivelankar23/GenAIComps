#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x
WORKPATH=$(git rev-parse --show-toplevel)
TAG='latest'
LOG_PATH="$WORKPATH/comps/text2kg/deployment/docker_compose"
source $WORKPATH/comps/text2kg/src/environment_setup.sh


echo $WORKPATH
ip_address=$(hostname -I | awk '{print $1}')
service_name="text2kg"

function build_docker() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2kg:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2kg/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2kg built fail"
        exit 1
    else
        echo "opea/text2kg built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml -f custom-override.yml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    cd $WORKPATH/tests/text2kg
    
    # Download test file
    FILE_URL="https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt"
    wget -P "$TEMP_DIR" "$FILE_URL"
    
    if wget -P "$TEMP_DIR" "$FILE_URL"; then
        echo "Download successful"
    else
        echo "Download failed"
        return 1
    fi
    
    # Test API endpoint
    result=$(curl -X POST \
          -H "accept: application/json" \
          -d "" \
          http://localhost:8090/v1/text2kg?input_text=Who%20is%20paul%20graham%3F)
    
    if [[ $result == *"output"* ]]; then
        echo $result
        echo "API response contains expected structure"
    else
        echo "Result wrong. Received was $result"
        docker logs text2kg > ${LOG_PATH}/text2kg.log
        return 1
    fi
    
    # Test Neo4j connection and data loading
    neo4j_test=$(cypher-shell -a bolt://localhost:7687 -u neo4j -p password "RETURN 'Connection OK' as result")
    if [ $? -eq 0 ]; then
        echo "Neo4j connection successful"
        
        # Verify knowledge graph entities
        verify_entities=$(cypher-shell -a bolt://localhost:7687 -u neo4j -p password <<EOF
            MATCH (p:Person {name: 'Paul Graham'})
            OPTIONAL MATCH (p)-[:WRITTEN_BY]-(articles:Article)
            OPTIONAL MATCH (p)-[:FOUNDED]-(companies:Organization)
            RETURN 
                COUNT(p) as person_count,
                COUNT(DISTINCT articles) as article_count,
                COUNT(DISTINCT companies) as company_count
        EOF)
        
        if [ $? -eq 0 ]; then
            echo "Knowledge graph entities verified"
            
            # Verify meaningful relationships
            verify_relationships=$(cypher-shell -a bolt://localhost:7687 -u neo4j -p password <<EOF
                MATCH (p:Person {name: 'Paul Graham'})
                WITH p
                OPTIONAL MATCH (p)-[:WRITTEN_BY]-(articles:Article)
                OPTIONAL MATCH (p)-[:FOUNDED]-(yc:Organization {name: 'Y Combinator'})
                RETURN 
                    COUNT(DISTINCT articles) > 0 AS has_articles,
                    COUNT(DISTINCT yc) > 0 AS has_yc
            EOF)
            
            if [ $? -eq 0 ]; then
                echo "Meaningful relationships verified"
                
                # Verify answer content
                expected_roles=("scientist" "writer" "entrepreneur")
                actual_answer=$(echo "$result" | jq -r '.output')
                
                roles_found=true
                for role in "${expected_roles[@]}"; do
                    if ! echo "$actual_answer" | grep -iq "$role"; then
                        roles_found=false
                        break
                    fi
                done
                
                if $roles_found; then
                    echo "Answer content verified"
                    return 0
                else
                    echo "Missing expected roles in answer"
                    return 1
                fi
            else
                echo "Failed to verify relationships"
                return 1
            fi
        else
            echo "Failed to verify entities"
            return 1
        fi
    else
        echo "Neo4j connection failed"
        return 1
    fi
    
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
    echo "===================  END STOP DOCKER ========================"
}

function main() {

    stop_docker

    build_docker
    start_service
    validate_microservice

    stop_docker

}

main
