Shader "Hidden/RVM/Composite"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
        _MaskTex ("Mask", 2D) = "white" {}
        _BackgroundTex ("Background", 2D) = "white" {}
        _BackgroundColor ("Background Color", Color) = (0, 1, 0, 1)
        _AlphaThreshold ("Alpha Threshold", Float) = 0.1
        _Feather ("Feather", Float) = 0.05
        _EdgeStrength ("Edge Strength", Float) = 0.5
        _UseBackgroundTex ("Use Background Texture", Float) = 0
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            sampler2D _MaskTex;
            sampler2D _BackgroundTex;
            float4 _MaskTex_TexelSize; // Unity auto-populates this
            float4 _BackgroundColor;
            float _AlphaThreshold;
            float _Feather;
            float _EdgeStrength;
            float _UseBackgroundTex;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Edge detection helper (matches compute shader)
            float getMaskEdgeFactor(float2 uv)
            {
                float2 texelSize = _MaskTex_TexelSize.xy;
                
                float l = tex2D(_MaskTex, uv + float2(-texelSize.x, 0)).r;
                float r = tex2D(_MaskTex, uv + float2(texelSize.x, 0)).r;
                float t = tex2D(_MaskTex, uv + float2(0, -texelSize.y)).r;
                float b = tex2D(_MaskTex, uv + float2(0, texelSize.y)).r;
                
                float gradient = abs(r - l) + abs(b - t);
                return saturate(gradient * 2.0);
            }

            // Edge refinement (matches compute shader)
            float refineMaskEdge(float mask, float edgeFactor, float strength)
            {
                if (edgeFactor < 0.01 || strength < 0.01) return mask;
                
                float refined = mask;
                if (mask > 0.5)
                    refined = mask + (1.0 - mask) * edgeFactor * strength;
                else
                    refined = mask * (1.0 - edgeFactor * strength);
                
                return saturate(refined);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float4 source = tex2D(_MainTex, i.uv);
                float mask = tex2D(_MaskTex, i.uv).r;
                
                // Apply edge refinement (matches compute shader logic)
                float finalAlpha = mask;
                if (_EdgeStrength > 0.001)
                {
                    float edgeFactor = getMaskEdgeFactor(i.uv);
                    finalAlpha = refineMaskEdge(mask, edgeFactor, _EdgeStrength);
                }
                
                // Apply threshold/feather only when edge refinement is OFF (matches compute shader)
                if (_EdgeStrength <= 0.001)
                {
                    float lower = max(0.0, _AlphaThreshold - _Feather);
                    float upper = min(1.0, _AlphaThreshold + _Feather);
                    float featherRange = upper - lower;
                    
                    if (featherRange > 0.001)
                    {
                        if (mask <= lower)
                            finalAlpha = 0.0;
                        else if (mask >= upper)
                            finalAlpha = 1.0;
                        else
                            finalAlpha = (mask - lower) / featherRange;
                    }
                    else
                    {
                        finalAlpha = mask >= _AlphaThreshold ? 1.0 : 0.0;
                    }
                }
                
                // Get background
                float4 bg = _UseBackgroundTex > 0.5 ? tex2D(_BackgroundTex, i.uv) : _BackgroundColor;
                
                // Alpha blending (matches compute shader)
                float3 blended = source.rgb * finalAlpha + bg.rgb * (1.0 - finalAlpha);
                return float4(blended, finalAlpha);
            }
            ENDCG
        }
    }
}
