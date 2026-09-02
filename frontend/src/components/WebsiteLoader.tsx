import React from 'react';
import styles from './WebsiteLoader.module.css';

export default function WebsiteLoader({ isFlyingOff = false }: { isFlyingOff?: boolean }) {
  return (
    <div className={`${styles.loaderContainer} ${isFlyingOff ? styles.fadeOutBg : ''}`}>
      <div className={styles.clouds}>
        <div className={`${styles.cloud} ${styles.cloud1}`}></div>
        <div className={`${styles.cloud} ${styles.cloud2}`}></div>
        <div className={`${styles.cloud} ${styles.cloud3}`}></div>
        <div className={`${styles.cloud} ${styles.cloud4}`}></div>
        <div className={`${styles.cloud} ${styles.cloud5}`}></div>
      </div>

      <div className={`${styles.loaderWrapper} ${isFlyingOff ? styles.flyOff : ''}`}>
        <div className={styles.loader}>
          <span><span></span><span></span><span></span><span></span></span>
          <div className={styles.base}>
            <span></span>
            <div className={styles.face}></div>
          </div>
        </div>
      </div>

      <div className={styles.longfazers}>
        <span></span><span></span><span></span><span></span>
      </div>
    </div>
  );
}
