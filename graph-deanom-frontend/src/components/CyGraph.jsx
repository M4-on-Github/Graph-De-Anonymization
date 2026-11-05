
import { useEffect, useRef } from 'react';
import cytoscape from '../../cytoscape';

function CyGraph() {
  const cyRef = useRef(null);

  useEffect(() => {
    const cy = cytoscape({
      container: cyRef.current,

      elements: [
        { data: { id: 'a', label: 'Node A' } },
        { data: { id: 'b', label: 'Node B' } },
        { data: { id: 'ab', source: 'a', target: 'b' } }
      ],

      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'background-color': '#007bff',
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 60,
            'height': 60
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#999',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#999',
            'curve-style': 'bezier'
          }
        }
      ],

      layout: {
        name: 'grid',
        fit: true,
        padding: 30
      }
    });

    // Cleanup when component unmounts
    return () => {
      cy.destroy();
    };
  }, []);

  return (
    <div>
      <h1>Graph Datasets</h1>
      <div
        ref={cyRef}
        style={{
          width: '100%',
          height: '600px',
          border: '1px solid #ddd'
        }}
      />
    </div>
  );
}

export default CyGraph;
