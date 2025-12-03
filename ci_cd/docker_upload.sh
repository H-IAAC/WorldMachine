#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"
donodo push --templates donodo_template.json eltoncn/world-machine:$1